import torch
import torch.fx
import operator

# Does this path not exist? Check that you've done the following:
# 1) Read README.md and follow the instructions to build libinterpreter.
# 2) If this file still does not exist after you've followed those instructions,
#    check if it is under a different extension (e.g. `dylib` on mac or `dll` on
#    windows).
torch.classes.load_library('build/libinterpreter.so')

# This is what a lowering pass should look like: a function that takes
# a valid nn.Module, symbolically traces it, lowers the Module to some
# representation, and wraps that representation up into another
# nn.Module instance that handles dispatch to the compiled/lowered code.
# This will ensure that this lowering transformation still fits into the
# PyTorch programming model and enables features like composing with other
# transformations and TorchScript compilation.
def lower_to_elementwise_interpreter(orig_mod : torch.nn.Module) -> torch.nn.Module:
    # ===== Stage 1: Symbolic trace the module =====
    mod = torch.fx.symbolic_trace(orig_mod)

    # ===== Stage 2: Lower GraphModule representation to the C++
    #       interpreter's instruction format ======
    instructions = []
    constant_idx = 0
    constants = {}
    fn_input_names = []

    target_to_name = {
        operator.add : "add",
        operator.mul : "mul"
    }

    output_node : Optional[torch.fx.Node] = None
    # For each instruction, create a triple
    # (instruction_name : str, inputs : List[str], output : str)
    # to feed into the C++ interpreter
    for n in mod.graph.nodes:
        target, args, out_name = n.target, n.args, n.name
        assert len(n.kwargs) == 0, "kwargs currently not supported"

        if n.op == 'placeholder':
            # Placeholders specify function argument names. Save these
            # for later when we generate the wrapper GraphModule
            fn_input_names.append(target)
        elif n.op == 'call_function':
            assert target in target_to_name, "Unsupported call target " + target
            arg_names = []
            for arg in args:
                if not isinstance(arg, torch.fx.Node):
                    # Pull out constants. These constants will later be
                    # fed to the interpreter C++ object via add_constant()
                    arg_name = f'constant_{constant_idx}'
                    constants[arg_name] = torch.Tensor(
                        [arg] if isinstance(arg, numbers.Number) else arg)
                    arg_names.append(arg_name)
                    constant_idx += 1
                else:
                    arg_names.append(arg.name)
            instructions.append((target_to_name[target], arg_names, out_name))
        elif n.op == 'output':
            if output_node is not None:
                raise RuntimeError('Multiple output nodes!')
            output_node = n
        else:
            raise RuntimeError('Unsupported opcode ' + n.op)

    interpreter = torch.classes.NativeInterpretation.ElementwiseInterpreter()
    # Load constants
    for k, v in constants.items():
        interpreter.add_constant(k, v)
    # Specify names for positional input arguments
    interpreter.set_input_names(fn_input_names)
    # Load instructions
    interpreter.set_instructions(instructions)
    # Specify name for single output
    assert isinstance(output_node.args[0], torch.fx.Node)
    interpreter.set_output_name(output_node.args[0].name)

    # ===== Stage 3: Create a wrapper GraphModule around the interpreter =====
    class WrapperModule(torch.nn.Module):
        def __init__(self, interpreter):
            super().__init__()
            self.interpreter = interpreter

    wrapper = WrapperModule(interpreter)

    # Create a forward() function that is compatible with TorchScript compilation.
    # Create a graph that: 1) Takes function arguments 2) Invokes the interpreter
    # 3) Returns the specified return value

    graph = torch.fx.Graph()
    # Add placeholders for fn inputs
    placeholder_nodes = []
    for name in fn_input_names:
        placeholder_nodes.append(graph.create_node('placeholder', name))

    # Get the interpreter object
    interpreter_node = graph.create_node('get_attr', 'interpreter')

    # Add a node to call the interpreter instance
    output_node = graph.create_node(
        op='call_method', target='__call__', args=(interpreter_node, placeholder_nodes))

    # Register output
    graph.output(output_node)

    graph.lint(wrapper)

    # Return final GraphModule!!!
    return torch.fx.GraphModule(wrapper, graph)

class MyElementwiseModule(torch.nn.Module):
    def forward(self, x, y):
        return x * y + y

mem = MyElementwiseModule()
lowered = lower_to_elementwise_interpreter(mem)
print(lowered.code)
# The lowered module can also be compiled into TorchScript
scripted = torch.jit.script(lowered)
print(scripted.graph)

# Stress test correctness
for _ in range(50):
    x, y = torch.randn(10, 20, 30), torch.randn(10, 20, 30)
    torch.testing.assert_allclose(lowered(x, y), mem(x, y))
    torch.testing.assert_allclose(scripted(x, y), mem(x, y))
