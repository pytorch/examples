import logging
from typing import Any, Callable, Dict, List, Optional

import torch
import torch.nn as nn
import torch.optim as optim
import torchdynamo
from functorch.compile import aot_module
from graph_profiler import GraphProfiler, GraphType
from torch import fx

from torchbenchmark.util.benchmark_utils import get_benchmark_model

FORWARD = GraphType.forward
BACKWARD = GraphType.backward


class ProfileEngine:
    r"""
    Obtain the forward pass and backward pass of the provided nn.Module and profile them. It provides the run function which takes
    an optional argument for running warm-up iterations before doing the actual profiling. Provides a print summary meothod to display
    node-wise profiling information in a tabulated manner. Provides acccess to the GraphProfiler object to use any node-wise profiling
    information for perfroming any transformations, rewriting or optimizations on the graph.

    Args:
        model (nn.Module): a local model instance of nn.Module.
        forward_loss (Callable): a function that takes and nn.module and input. It calls the model with the provided inputs and
                                calculates and returns the loss.
        optimizer (optim.Optimizer) : an instance of model's registered optimizer. The Optimizer is needed for accounting for
                                    the memory occupied by the optimizer states and parameter grads. Currently optimizer is not
                                     traceable and hence it won't be a part of the graph profiling mechanism.
        example_inputs (Any): The example inputs will be passed to the forward_loss function to obtain the forward pass and
                                loss of the model.
        profile_mode (str): The Graph Profiler provides three profiling modes,``default``, ``swap`` and ``mem_saver_swap``.
                            default: Measure the per node run-time, active memory usage, peak memory usage, marks the intermediate
                                    nodes (activations saved from forward pass and needed in backward pass), registers their last
                                    use in the forward pass and first use in the backward pass, measures this idle time, measures
                                    their memory size and, marks the first use of the model parameter nodes in the forward pass.
                            swap:   All the of the above plus measures the time to swap each of the intermediate tensors (activations)
                                     to CPU memory, back and forth.
                            mem_saver_swap: All of the above and profiles in a low memory mode, pushing all of the activations to
                                            the CPU memory during the forward pass and fetches them back when they are needed in
                                            the backward pass. Allows profiling graphs way larger than GPU memory.
    """

    def __init__(
        self,
        model: nn.Module,
        forward_loss: Callable,
        optimizer: optim.Optimizer,
        example_inputs: Any,
        profile_mode: str,
    ):

        self.model: nn.Module = model
        self.forward_loss = forward_loss
        self.optimizer: optim.Optimizer = optimizer
        self.example_inputs: Any = example_inputs
        self.profile_mode: str = profile_mode
        self.profile_ctx = None
        self.profilers: Dict[int, Dict[GraphType, GraphProfiler]] = {}

    def run(self, warm_up_iters: Optional[int] = 0, profile_iters:Optional[int]=1):
        r"""
        Calls the _compile method to initialize the profiler context. Runs optional warm-up profiling iterations. 
        This is sometimes essential to warm-up the cuda caching allocator and initilize the pinned CPU memory when
        profiling for swapping times as well. Subsequent to warm-up, all the profiler statistics are reset and the 
        actual profiling is done for number of iterations specified.
        Args:
            warmp_up_iters (int): Optional number of warm-up iterations to perform. Default: 0
            profile_iters (int): Number of profiling iterations to perform. Default: 1
        """
        if self.profile_ctx is None:
            self.profile_ctx = self._compile()

        logging.info("Warming up profiler...")
        for _ in range(warm_up_iters):
            with self.profile_ctx:
                self.forward_loss(self.model, self.example_inputs).backward()
            self.optimizer.zero_grad(False)

        self.reset_stats()

        logging.info("Profling...")
        for _ in range(profile_iters):
            with self.profile_ctx:
                self.forward_loss(self.model, self.example_inputs).backward()
            self.optimizer.zero_grad(False)

    def _summary(self) -> None:
        r"""
        Aggregates all the statistics accumulated during the profiling iterations and makes them ready for printing.
        """
        for prof_dict in self.profilers.values():
            fwd_profiler: GraphProfiler = prof_dict[FORWARD]
            bwd_profiler: GraphProfiler = prof_dict[BACKWARD]
            fwd_profiler.summarize()
            bwd_profiler.summarize()

    def reset_stats(self):
        r"""
        Resets all the accumulated profiling statistics. Usualy called after warm-up iterations or before beginning 
        a new profiling session.
        """
        for prof_dict in self.profilers.values():
            fwd_profiler: GraphProfiler = prof_dict[FORWARD]
            bwd_profiler: GraphProfiler = prof_dict[BACKWARD]
            fwd_profiler.reset_stats()
            bwd_profiler.reset_stats()

    def print_summary(self) -> str:
        r"""
        Calls the summarize method for all the profilers and prints the profiling statistics for all the forward and 
        backward graphs captured during initialization.
        """
        self._summary()
        for gid, prof_dict in self.profilers.items():
            fwd_profiler: GraphProfiler = prof_dict[FORWARD]
            logging.info(f"Forward Graph {gid} Summary: ")
            print(fwd_profiler.print_summary())
            bwd_profiler: GraphProfiler = prof_dict[BACKWARD]
            logging.info(f"Backward Graph {gid} Summary: ")
            print(bwd_profiler.print_summary())

    def _aot_compile_fwd(self, dynamo_fwd_gm: fx.GraphModule):
        # Wraps the forward compiler for the aot_module. 
        # 1) It initializes the forward graph profiler.
        # 2) Stores the reference of the profiler with the corresponding graph_id.
        # 3) Plugs-in the profiler's run method as a callable to the forward pass.
        def compile_fwd(gm: fx.GraphModule, inps) -> fx.GraphModule:
            nonlocal dynamo_fwd_gm
            logging.info(f"Compiling Forward Graph: {dynamo_fwd_gm._id}")
            # print(gm.graph)
            fwd_profiler: GraphProfiler = GraphProfiler(
                gm,
                FORWARD,
                fw_num_outs=dynamo_fwd_gm._num_outs,
                sync=False,
                profile_mode=self.profile_mode,
            )
            self.profilers[dynamo_fwd_gm._id][FORWARD] = fwd_profiler
            return fwd_profiler.run

        return compile_fwd

    def _aot_compile_bwd(self, dynamo_fwd_gm: fx.GraphModule):
        # Wraps the backward compiler for the aot_module. 
        # 1) It initializes the backward graph profiler using the corresponding forward profiler.
        # 2) Stores the reference of the profiler with the corresponding graph_id.
        # 3) Plugs-in the profiler's ``meta_run`` method as a callable to the backward pass.
        # NOTE: The meta_run method is plugged for the backward pass due to difference in the way 
        # arguments are passed to the forward and backward passes to address the memory release issue.
        def compile_bwd(gm: fx.GraphModule, inps) -> fx.GraphModule:
            nonlocal dynamo_fwd_gm
            logging.info(f"Compiling Backward Graph: {dynamo_fwd_gm._id}")
            # print(gm.graph)
            fwd_profiler: GraphProfiler = self.profilers[dynamo_fwd_gm._id][FORWARD]
            bwd_profiler: GraphProfiler = GraphProfiler(
                gm, BACKWARD, fwd_profiler=fwd_profiler
            )
            self.profilers[dynamo_fwd_gm._id][BACKWARD] = bwd_profiler
            return bwd_profiler.meta_run

        return compile_bwd

    def _compile(self):
        r"""
        Runs the self.forward_loss callable with self.model and self.example_inputs under torchdynamo context with a 
        custom compiler (dynamo_compiler). The dynamo compiler extracts each sub-graph from the context, counts the 
        number of outputs in the graph, records the unique graph id and calls aot_module, with custom forward and
        backward compilers where the profilers for them are initialized and plugged in.
        """
        gid = 0

        def dynamo_compiler(gm: fx.GraphModule, example_inputs: List[torch.Tensor]):
            nonlocal gid
            logging.info(f"Compiling Dynamo Graph: {gid}")
            output_count = 0
            for node in gm.graph.nodes:
                if node.op == "output":
                    for _ in node.args[0]:
                        output_count += 1
            gm._id, gm._num_outs = gid, output_count
            self.profilers[gid] = {}
            compiled_m = aot_module(
                gm, self._aot_compile_fwd(gm), self._aot_compile_bwd(gm)
            )
            gid += 1
            return compiled_m

        optimize_ctx = torchdynamo.optimize(dynamo_compiler)
        with optimize_ctx:
            self.forward_loss(self.model, self.example_inputs)

        return optimize_ctx


if __name__ == "__main__":

    logging.getLogger().setLevel(logging.DEBUG)

    model_name = "torchbenchmark.models.hf_Bert.Model"
    batch_size = 16
    device = torch.device("cuda")
    model, forward_loss, optimizer, example_inputs = get_benchmark_model(
        model_name, batch_size=batch_size, device=device
    )

    engine = ProfileEngine(model, forward_loss, optimizer, example_inputs, "default")
    engine.run(warm_up_iters=2)
    engine.print_summary()
