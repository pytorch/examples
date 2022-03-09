# This example is provided only for explanatory and educational purposes. The
# underlying APIs may change and this tutorial may break.

# Compiling FX models to NNC (Neural Network Compiler)
######################################################
# The goal of this file is to demonstrate an end to end example of using FX to
# lower a PyTorch model to a backend codegen compiler. In this example, we will
# be using NNC
# (https://github.com/pytorch/pytorch/blob/master/test/cpp/tensorexpr/tutorial.cpp).
# If you're unfamiliar with NNC, the general design is strongly inspired by TVM
# and Halide.
#
# To do so, this example contains two FX transformations.
# The first one is a decomposition pass that normalizes and decomposes PyTorch
# operations (such as addmm). Using a pass like this allows us to reduce the
# number of lowerings we need to write. Instead of needing to specifically
# write a lowering for addmm, we can decompose addmm and lower its constituent
# operations.
# The second one is the actual lowering pass itself. In this case, we will need
# to convert each PyTorch operation we encounter into the corresponding NNC
# `TensorExpr`.
#
# These two passes, `decompose` and `nnc_compile`, are fairly similar.
# In both cases, we re-interpret each operation in the FX graph to construct an
# entirely new representation. In the decomposition pass, we either copy the
# operation as-is into the new graph, or we use `Proxy` objects to decompose
# the operation. This is an extension of the example presented here:
# https://pytorch.org/docs/master/fx.html#proxy-retracing
#
# In the lowering pass, a similar principle applies. However, instead of using
# `Proxy` objects to rewrite our op in other PyTorch ops, we do the translation
# ourselves. In addition, since this is not a source-to-source transformation,
# we return a somewhat hacky function that passes in the module attributes to
# the NNC callable.
#
# Results
######################################
# Using NNC (which compiles directly to LLVM), we can compile a fairly small
# PyTorch model and compare performnance between NNC, PyTorch Eager, and Static
# Runtime. These are my resuls on an Intel i7-8750H CPU.
#
# NNC time:  0.0066373348236083984
# PyTorch time 0.025979042053222656
# Static Runtime time 0.011004209518432617
#
# As we can see, NNC is nearly 2x faster than static runtime and more than 4x
# faster than PyTorch. This is not surprising, as we are dealing with extremely
# small tensors where framework overhead is a significant factor.

import time
import torch
import torch.nn as nn
import torch._C._te as te
import torch.fx as fx
from torch.fx import map_arg
from torch.fx.passes.shape_prop import ShapeProp
import operator

# Decomposition Pass

def binary_mapping(op):
    def f(a, b):
        return op(a, b)
    return f

decomposition_rules = {}
binary_decompositions = [
    (operator.matmul, torch.mm),
    (operator.add, torch.add),
    (operator.mul, torch.mul),
    (operator.sub, torch.sub),
    (operator.truediv, torch.div),
    (operator.eq, torch.eq),
    (operator.gt, torch.gt),
    (operator.ge, torch.ge),
    (operator.lt, torch.lt),
    (operator.le, torch.le),
    (operator.ne, torch.ne),
    (operator.and_, torch.bitwise_and)
]
for old, new in binary_decompositions:
    decomposition_rules[old] = binary_mapping(new)

def addmm_decompose(input, mat1, mat2, beta=1, alpha=1, out=None):
    assert(out is None)
    return beta*input + alpha*(torch.mm(mat1, mat2))

decomposition_rules[torch.addmm] = addmm_decompose

def decompose(model: torch.nn.Module, example_inputs) -> torch.nn.Module:
    """
    decompose(model, example_inputs) takes in a model, decomposes any of the functions in `decomposition_rules` to its constituent operations, and returns a `nn.Module` without any of the operations with decomposition rules.
    """
    # Run it multiple times so we converge to a fixed point.
    for _ in range(5):
        model = fx.symbolic_trace(model)
        ShapeProp(model).propagate(*example_inputs)
        new_graph = fx.Graph()
        env = {}
        for node in model.graph.nodes:
            if node.op == 'call_function' and node.target in decomposition_rules:
                # If the current function is in `decomposition_rules`, we use
                # `Proxy` objects to decompose the operations using the
                # decomposition rule. See
                # https://pytorch.org/docs/master/fx.html#proxy-retracing for
                # more details.
                proxy_args = map_arg(node.args, lambda n: fx.Proxy(env[n.name]))
                proxy_kwargs = map_arg(node.kwargs, lambda n: fx.Proxy(env[n.name]))
                new_node = decomposition_rules[node.target](*proxy_args, **proxy_kwargs).node
                env[node.name] = new_node
            else:
                new_node = new_graph.node_copy(node, lambda x: env[x.name])
                env[node.name] = new_node
        model = fx.GraphModule(model, new_graph)
    return model

# NNC Lowering Pass

class kernel_arena_scope(object):
    def __enter__(self):
        self.scope = te.KernelScope()

    def __exit__(self, typ, val, traceback):
        self.scope = None

def get_dim_args(dims):
    dim_args = []
    for dim in dims:
        dim_args.append(te.DimArg(te.ExprHandle.int(dim), 'i' + str(len(dim_args))))
    return dim_args

def to_expr(x):
    if isinstance(x, int):
        return te.ExprHandle.int(x)
    elif isinstance(x, float):
        return te.ExprHandle.float(x)


lowering_functions = {}

def wrap_compute(f):
    def fn_lower(name, out_shape, inp_shapes, args):
        X = te.Compute(name, get_dim_args(out_shape), f(inp_shapes, args))
        return X
    return fn_lower

def gen_unary_nnc(op):
    def gen_op_nnc(inp_shapes, args):
        def f(*idxs):
            return op(args[0].load(idxs))
        return f
    return gen_op_nnc

unary_lowerings = [
    (torch.sin, lambda x: x.sin()),
    (torch.cos, lambda x: x.cos()),
    (torch.tan, lambda x: x.tan()),
    (torch.asin, lambda x: x.asin()),
    (torch.acos, lambda x: x.acos()),
    (torch.atan, lambda x: x.atan()),
    (torch.sinh, lambda x: x.sinh()),
    (torch.cosh, lambda x: x.cosh()),
    (torch.tanh, lambda x: x.tanh()),
    (torch.sigmoid, lambda x: x.sigmoid()),
    (torch.exp, lambda x: x.exp()),
    (torch.expm1, lambda x: x.expm1()),
    (torch.expm1, lambda x: x.expm1()),
    (torch.abs, lambda x: x.abs()),
    (torch.log, lambda x: x.log()),
    (torch.log2, lambda x: x.log2()),
    (torch.log10, lambda x: x.log10()),
    (torch.log1p, lambda x: x.log1p()),
    (torch.erf, lambda x: x.erf()),
    (torch.erfc, lambda x: x.erfc()),
    (torch.sqrt, lambda x: x.sqrt()),
    (torch.rsqrt, lambda x: x.rsqrt()),
    (torch.ceil, lambda x: x.ceil()),
    (torch.floor, lambda x: x.floor()),
    (torch.round, lambda x: x.round()),
    (torch.trunc, lambda x: x.trunc()),
    (torch.lgamma, lambda x: x.lgamma()),
]

for torch_op, nnc_fn in unary_lowerings:
    lowering_functions[torch_op] = wrap_compute(gen_unary_nnc(nnc_fn))

def gen_binary_nnc(op):
    def is_nnc_obj(x):
        return isinstance(x, te.Placeholder) or isinstance(x, te.Tensor)
    def gen_op_nnc(inp_shapes, args):
        if is_nnc_obj(args[0]) and is_nnc_obj(args[1]):
            A_shape, A_dtype = inp_shapes[0]
            B_shape, B_dtype = inp_shapes[1]
            A, B = args

            def index_or_broadcast(shape, *args):
                out = []
                for idx, arg in enumerate(args):
                    if idx >= len(shape): continue
                    if shape[idx] == 1:
                        out.append(to_expr(0))
                    else:
                        out.append(arg)
                return out

            def f(*idxs):
                return op(A.load(index_or_broadcast(A_shape, *idxs)), B.load(index_or_broadcast(B_shape, *idxs)))
            return f
        else:
            if is_nnc_obj(args[0]):
                def f(*idxs):
                    return op(args[0].load(idxs), to_expr(args[1]))
                return f
            else:
                def f(*idxs):
                    return op(to_expr(args[0]), args[1].load(idxs))
                return f

    return gen_op_nnc


binary_lowerings = [
(torch.add,lambda a, b: a+b),
(torch.mul,lambda a, b: a*b),
(torch.sub,lambda a, b: a-b),
(torch.div,lambda a, b: a/b),
(torch.eq,lambda a, b: a==b),
(torch.gt,lambda a, b: a>b),
(torch.lt,lambda a, b: a<b),
(torch.ge,lambda a, b: a>=b),
(torch.le,lambda a, b: a<=b),
]
for torch_op, nnc_fn in binary_lowerings:
    lowering_functions[torch_op] = wrap_compute(gen_binary_nnc(nnc_fn))

def clamp_lower(inp_shapes, args):
    def f(*idxs):
        val = args[0].load(idxs)
        return te.ifThenElse(val < to_expr(args[1]), to_expr(args[1]),
                            te.ifThenElse(val > to_expr(args[2]), to_expr(args[2]), val))
    return f

lowering_functions[torch.clamp] = wrap_compute(clamp_lower)

def transpose_lower(name, out_shape, inp_shapes, args):
    idx_1, idx_2 = args[1], args[2]
    def transpose(shape):
        shape[idx_1], shape[idx_2] = shape[idx_2], shape[idx_1]
        return shape
    def f(*idxs):
        idxs = transpose(list(idxs))
        return args[0].load(idxs)
    return te.Compute(name, get_dim_args(out_shape), f)

def flatten_lower(name, out_shape, inp_shapes, args):
    A, start_dim, end_dim = args
    shape = list(inp_shapes[0][0])
    flattened_region = shape[start_dim:end_dim+1]
    def prod(x):
        t = 1
        for i in x:
            t *= i
        return t
    def get_orig_idxs(i):
        idxs = []
        total = prod(flattened_region)
        for dim in flattened_region:
            total //= dim
            idxs.append(i / to_expr(total))
            i = i % to_expr(total)
        return idxs
    def f(*idxs):
        idxs = list(idxs)
        idxs = idxs[:start_dim] + get_orig_idxs(idxs[start_dim]) + idxs[start_dim+1:]
        return A.load(idxs)
    return te.Compute(name, get_dim_args(out_shape), f)

def cat_lower(name, out_shape, inp_shapes, args):
    tensors = args[0]
    dim = args[1]
    lengths = [i[0][dim] for i in inp_shapes[0]]
    def f(*idxs):
        idxs = list(idxs)
        sm = lengths[0]
        load = tensors[0].load(idxs)
        for length, tensor in list(zip(lengths, tensors))[1:]:
            new_idxs = idxs[:]
            new_idxs[dim] -= to_expr(sm)
            load = te.ifThenElse(idxs[dim] < to_expr(sm), load, tensor.load(new_idxs))
        return load
    return te.Compute(name, get_dim_args(out_shape), f)

lowering_functions[torch.transpose] = transpose_lower
lowering_functions[torch.flatten] = flatten_lower
lowering_functions[torch.cat] = cat_lower

def bmm_lower(name, out_shape, inp_shapes, args):
    M1 = args[0]
    M2 = args[1]
    B, N, M = inp_shapes[0][0]
    P = inp_shapes[1][0][2]

    def f(b, n, p, m):
        return M1.load([b, n, m]) * M2.load([b, m, p])
    mm = te.Compute('mm', get_dim_args([B,N,P,M]), f)
    return te.Reduce(name, get_dim_args([B, N, P]), te.Sum(), mm, get_dim_args([M]))


def mm_lower(name, out_shape, inp_shapes, args):
    M1 = args[0]
    M2 = args[1]
    N, M = inp_shapes[0][0]
    P = inp_shapes[1][0][1]

    def f(n, p, m):
        return M1.load([n, m]) * M2.load([m, p])
    mm = te.Compute('mm', get_dim_args([N,P,M]), f)
    return te.Reduce(name, get_dim_args([N, P]), te.Sum(), mm, get_dim_args([M]))

lowering_functions[torch.bmm] = bmm_lower
lowering_functions[torch.mm] = mm_lower


def lower_function(node, op, nnc_args, args):
    inp_shapes = fx.node.map_aggregate(args, lambda arg: (arg.shape, arg.dtype) if isinstance(arg, fx.Node) else None)
    return lowering_functions[op](node.name, node.shape, inp_shapes, nnc_args)

def nnc_compile(model: torch.nn.Module, example_inputs) -> torch.nn.Module:
    """
    nnc_compile(model, example_inputs) returns a function with the same args
    as `model.forward`, with an extra argument corresponding to where the
    output is stored. This function takes the inputs (which must be PyTorch
    tensors with the same shapes as example_inputs), and passes them to an
    NNC executor.
    """
    fx_model = fx.symbolic_trace(model)
    ShapeProp(fx_model).propagate(*example_inputs)

    # This env maps from nodes to `te.ExprHandle`, which represent the output
    # of an NNC computation.
    env = {}
    def get_te_shapes(node):
        return [te.ExprHandle.int(i) for i in node.shape]

    def get_nnc_type(dtype):
        if dtype == torch.float:
            return te.Dtype.Float
        elif dtype == torch.long:
            return te.Dtype.Long
        else:
            raise RuntimeError("nyi")

    def get_te_type(node):
        return get_nnc_type(node.dtype)

    def gen_compute(args):
        te_args = [env[arg.name] for arg in args]

    def lookup_env(l):
        return fx.node.map_aggregate(l, lambda x: env[x.name] if isinstance(x, fx.Node) else x)

    def fetch_attr(target : str):
        target_atoms = target.split('.')
        attr_itr = fx_model
        for i, atom in enumerate(target_atoms):
            if not hasattr(attr_itr, atom):
                raise RuntimeError(f"Node referenced nonexistant target {'.'.join(target_atoms[:i])}")
            attr_itr = getattr(attr_itr, atom)
        return attr_itr

    outs = None
    inputs = []
    module_attrs = []
    for node in fx_model.graph.nodes:
        if node.op == 'placeholder':
            # We simply map the input placeholder to a `te.Placeholder`, which
            # also represents an input to the NNC computation.
            shapes = get_te_shapes(node)
            env[node.name] = te.Placeholder(node.name, get_te_type(node), shapes)
            inputs.append(env[node.name])
        elif node.op == 'call_function':
            # This does the bulk of the work - we call `lower_function`, which
            # returns a `te.ExprHandle` (the output of a NNC computation), and
            # put it in our environment.
            result = lower_function(node, node.target, lookup_env(node.args), node.args)
            env[node.name] = result
        elif node.op == 'output':
            outs = list(lookup_env(node.args))
        elif node.op == 'get_attr':
            # As NNC doesn't have any concept of state, we pull out the module
            # attributes and pass them in as inputs to NNC.
            module_attrs.append(node)
            env[node.name] = te.Placeholder(node.name, get_te_type(node), shapes)
        else:
            raise RuntimeError("not yet implemented")

    loopnest = te.LoopNest(outs)
    loopnest.prepare_for_codegen()
    stmt = te.simplify(loopnest.root_stmt())
    cg = te.construct_codegen('llvm', stmt, [te.BufferArg(x) for x in [env[i.name] for i in module_attrs] + inputs + outs])
    def f(inps):
        module_stuff = [fetch_attr(i.target) for i in module_attrs]
        cg.call(module_stuff + list(inps))
    return f


################################
# Example usage and Benchmarking
################################

if __name__ == '__main__':
    class DeepAndWide(torch.nn.Module):
        def __init__(self, num_features=50):
            super(DeepAndWide, self).__init__()
            self.mu = torch.nn.Parameter(torch.randn(1, num_features))
            self.sigma = torch.nn.Parameter(torch.randn(1, num_features))
            self.fc_w = torch.nn.Parameter(torch.randn(1, num_features + 1))
            self.fc_b = torch.nn.Parameter(torch.randn(1))

        def forward(self, ad_emb_packed, user_emb, wide):
            wide_offset = wide + self.mu
            wide_normalized = wide_offset * self.sigma
            wide_preproc = torch.clamp(wide_normalized, 0., 10.)
            user_emb_t = torch.transpose(user_emb, 1, 2)
            dp_unflatten = torch.bmm(ad_emb_packed, user_emb_t)
            dp = torch.flatten(dp_unflatten, 1, -1)
            inp = torch.cat([dp, wide_preproc], 1)
            t1 = torch.transpose(self.fc_w, 1, 0)
            fc1 = torch.addmm(self.fc_b, inp, t1)
            return fc1

    with kernel_arena_scope():
        with torch.no_grad():
            num_features = 50
            mod = DeepAndWide(num_features)

            # Phabricate sample inputs
            batch_size = 1
            embedding_size = 32
            ad_emb_packed = torch.randn(batch_size, 1, embedding_size)
            user_emb = torch.randn(batch_size, 1, embedding_size)
            wide = torch.randn(batch_size, num_features)
            inps = (ad_emb_packed, user_emb, wide)
            out = torch.empty(batch_size, 1)

            mod = decompose(mod, inps)
            cg = nnc_compile(mod, inps)

            iters = 1000

            for _ in range(10):
                cg([ad_emb_packed, user_emb,wide, out])
            begin = time.time()
            for _ in range(iters):
                cg([ad_emb_packed, user_emb,wide, out])

            print("NNC time: ", time.time()-begin)

            mod_jit = torch.jit.script(DeepAndWide(num_features))
            for _ in range(10):
                mod_jit(ad_emb_packed, user_emb,wide)
            begin = time.time()
            for _ in range(iters):
                mod_jit(ad_emb_packed, user_emb,wide)
            print("PyTorch time", time.time()-begin)

            static_runtime = torch._C._jit_to_static_runtime(mod_jit._c)
            for _ in range(10):
                static_runtime.run([ad_emb_packed, user_emb,wide])
            begin = time.time()
            for _ in range(iters):
                static_runtime.run([ad_emb_packed, user_emb,wide])
            print("Static Runtime time", time.time()-begin)

            print("Sums:", out.sum(), mod(*inps).sum())
