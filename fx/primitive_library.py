import torch
import torch.fx
"""
In this example we are going do define a library of
"composite" operations. Composite operations are those
that are defined as callable functions that are composed
of several other operations in their implementation.

Composite operations allow you to choose at what level
of abstraction you want to interpret/manipulate the
code. We show that we can provide a function to inline
these functions as well as use a custom Tracer to auto-
matically inline such functions.

Composite operations can be useful for exposing higher-
level context to a backend/transform while still
maintaining the ability to examine things at a more
fine-grained level.
"""


def sigmoid_lowp(x : torch.Tensor):
    x = x.float()
    x = x.sigmoid()
    return x.half()

# wrap() indicates that the passed-in function should always
# be recorded as a call_function node rather than being traced
# through. Later, we will see how we can:
# a. Inline the implementation of such a function and
# b. Define a tracer that automatically traces through such a function
torch.fx.wrap(sigmoid_lowp)

def add_lowp(a : torch.Tensor, b : torch.Tensor):
    a, b = a.float(), b.float()
    c = a + b
    return c.half()

torch.fx.wrap(add_lowp)


# Let's see what happens when we symbolically trace through some code
# that uses these functions

class Foo(torch.nn.Module):
    def forward(self, x, y):
        x = sigmoid_lowp(x)
        y = sigmoid_lowp(y)
        return add_lowp(x, y)


traced = torch.fx.symbolic_trace(Foo())
print(traced.code)
"""
def forward(self, x, y):
    sigmoid_lowp = __main___sigmoid_lowp(x);  x = None
    sigmoid_lowp_1 = __main___sigmoid_lowp(y);  y = None
    add_lowp = __main___add_lowp(sigmoid_lowp, sigmoid_lowp_1);  sigmoid_lowp = sigmoid_lowp_1 = None
    return add_lowp
"""

# Notice that the calls to `sigmoid_lowp` and `add_lowp`
# appear literally in the trace; they are not traced through


# ***** Inlining calls *****
# Now let's define a function that allows for inlining these calls
# during graph manipulation.

def inline_lowp_func(n : torch.fx.Node):
    # If we find a call to a function in our "lowp" module, inline it
    if n.op == 'call_function' and n.target.__module__ == inline_lowp_func.__module__:
        # We want to insert the operations comprising the implementation of the
        # function before the function itself. Then, we can swap the output value
        # of the function call with the output value for its implementation nodes
        tracer = torch.fx.proxy.GraphAppendingTracer(n.graph)
        with n.graph.inserting_before(n):
            # We can inline code by using `fx.Proxy` instances.
            # map_arg traverses all aggregate types and applies the given function
            # to Node instances in the data structure. In this case, we are applying
            # the fx.Proxy constructor.
            proxy_args = torch.fx.node.map_arg(n.args, lambda x: torch.fx.Proxy(x, tracer))
            proxy_kwargs = torch.fx.node.map_arg(n.kwargs, lambda x: torch.fx.Proxy(x, tracer))
            # Call the function itself with proxy arguments. This will emit
            # nodes in the graph corresponding to the operations in the im-
            # plementation of the function
            output_proxy = n.target(*proxy_args, **proxy_kwargs)
            # Now replace the original node's uses with the output node of
            # the implementation.
            node.replace_all_uses_with(output_proxy.node)
            # Delete the old node
            node.graph.erase_node(node)

for node in traced.graph.nodes:
    if node.op == 'call_function' and node.target is sigmoid_lowp:
        inline_lowp_func(node)

# Don't forget to recompile after graph manipulation
traced.recompile()

print(traced.code)
"""
def forward(self, x, y):
    float_1 = x.float();  x = None
    sigmoid = float_1.sigmoid();  float_1 = None
    half = sigmoid.half();  sigmoid = None
    float_2 = y.float();  y = None
    sigmoid_1 = float_2.sigmoid();  float_2 = None
    half_1 = sigmoid_1.half();  sigmoid_1 = None
    add_lowp = __main___add_lowp(half, half_1);  half = half_1 = None
    return add_lowp
"""

# At this point, the implementation of `sigmoid_lowp` has been substituted
# in for all of the calls to that function.

# ***** Inlining calls during tracing *****
# Now we are going to define a custom tracer that can selectively inline
# calls to certain composite operations on-the-fly.

# New instance of our module
f = Foo()

class InliningTracer(torch.fx.Tracer):
    FNS_TO_INLINE = [add_lowp]

    def create_node(self, kind, target, args, kwargs, name=None, type_expr=None):
        if kind == 'call_function' and target in self.FNS_TO_INLINE:
            tracer = torch.fx.proxy.GraphAppendingTracer(self.graph)
            # Trace through the implementation of the function rather than
            # create a node
            proxy_args = torch.fx.node.map_arg(args, lambda x: torch.fx.Proxy(x, tracer))
            proxy_kwargs = torch.fx.node.map_arg(kwargs, lambda x: torch.fx.Proxy(x, tracer))
            return target(*proxy_args, **proxy_kwargs).node
        else:
            return super().create_node(kind, target, args, kwargs, name, type_expr)


tracer = InliningTracer()
graph = tracer.trace(f)
module = torch.fx.GraphModule(f, graph)
print(module.code)
"""
def forward(self, x, y):
    sigmoid_lowp = __main___sigmoid_lowp(x);  x = None
    sigmoid_lowp_1 = __main___sigmoid_lowp(y);  y = None
    float_1 = sigmoid_lowp.float();  sigmoid_lowp = None
    float_2 = sigmoid_lowp_1.float();  sigmoid_lowp_1 = None
    add = float_1 + float_2;  float_1 = float_2 = None
    half = add.half();  add = None
    return half
"""

# As you can see, the implementation for `add_lowp` has been
# inlined in the course of tracing with our InliningTracer.
# Such functionality can be used to, for example, implement
# a backend that wants to see the lowered form of some operations
# but the high-level form of another.

# ***** Future direction *****
#
# We may define an API, such as `Tracer.is_leaf_function`, that
# Tracer implementers can use to more easily specify the inlining
# behavior implemented in InliningTracer. Such a method would return
# True by default, but a Tracer can override it and return `False` for
# functions the Tracer wants to be traced through.
