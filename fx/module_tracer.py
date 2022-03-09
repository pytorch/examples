"""
Recording Module Hierarchy With a Custom Tracer

In this example, we are going to define a custom `fx.Tracer` instance that--
for each recorded operation--also notes down the qualified name of the module
from which that operation originated. The _qualified name_ is the path to the
Module from the root module. More information about this concept can be
found in the documentation for `Module.get_submodule`:
https://github.com/pytorch/pytorch/blob/9f2aea7b88f69fc74ad90b1418663802f80c1863/torch/nn/modules/module.py#L385
"""
import torch
import torch.fx
from typing import Any, Callable, Dict, Optional, Tuple

class ModulePathTracer(torch.fx.Tracer):
    """
    ModulePathTracer is an FX tracer that--for each operation--also records
    the qualified name of the Module from which the operation originated.
    """

    # The current qualified name of the Module being traced. The top-level
    # module is signified by empty string. This is updated when entering
    # call_module and restored when exiting call_module
    current_module_qualified_name : str = ''
    # A map from FX Node to the qualname of the Module from which it
    # originated. This is recorded by `create_proxy` when recording an
    # operation
    node_to_originating_module : Dict[torch.fx.Node, str] = {}

    def call_module(self, m: torch.nn.Module, forward: Callable[..., Any],
                    args : Tuple[Any, ...], kwargs : Dict[str, Any]) -> Any:
        """
        Override of Tracer.call_module (see
        https://pytorch.org/docs/stable/fx.html#torch.fx.Tracer.call_module).

        This override:
        1) Stores away the qualified name of the caller for restoration later
        2) Installs the qualified name of the caller in `current_module_qualified_name`
           for retrieval by `create_proxy`
        3) Delegates into the normal Tracer.call_module method
        4) Restores the caller's qualified name into current_module_qualified_name
        """
        old_qualname = self.current_module_qualified_name
        try:
            self.current_module_qualified_name = self.path_of_module(m)
            return super().call_module(m, forward, args, kwargs)
        finally:
            self.current_module_qualified_name = old_qualname

    def create_proxy(self, kind: str, target: torch.fx.node.Target, args: Tuple[Any, ...],
                     kwargs: Dict[str, Any], name: Optional[str] = None, type_expr: Optional[Any] = None):
        """
        Override of `Tracer.create_proxy`. This override intercepts the recording
        of every operation and stores away the current traced module's qualified
        name in `node_to_originating_module`
        """
        proxy = super().create_proxy(kind, target, args, kwargs, name, type_expr)
        self.node_to_originating_module[proxy.node] = self.current_module_qualified_name
        return proxy


# Testing: let's see how this works on a torchvision ResNet18 model
import torchvision.models as models

# Model under test
rn18 = models.resnet18()

# Instantiate our ModulePathTracer and use that to trace our ResNet18
tracer = ModulePathTracer()
traced_rn18 = tracer.trace(rn18)

# Print (node, module qualified name) for every node in the Graph
for node in traced_rn18.nodes:
    module_qualname = tracer.node_to_originating_module.get(node)
    print('Node', node, 'is from module', module_qualname)
"""
Node x is from module 
Node conv1 is from module conv1
Node bn1 is from module bn1
Node relu is from module relu
Node maxpool is from module maxpool
Node layer1_0_conv1 is from module layer1.0.conv1
Node layer1_0_bn1 is from module layer1.0.bn1
Node layer1_0_relu is from module layer1.0.relu
Node layer1_0_conv2 is from module layer1.0.conv2
Node layer1_0_bn2 is from module layer1.0.bn2
Node add is from module layer1.0
Node layer1_0_relu_1 is from module layer1.0.relu
Node layer1_1_conv1 is from module layer1.1.conv1
Node layer1_1_bn1 is from module layer1.1.bn1
Node layer1_1_relu is from module layer1.1.relu
Node layer1_1_conv2 is from module layer1.1.conv2
Node layer1_1_bn2 is from module layer1.1.bn2
Node add_1 is from module layer1.1
Node layer1_1_relu_1 is from module layer1.1.relu
Node layer2_0_conv1 is from module layer2.0.conv1
Node layer2_0_bn1 is from module layer2.0.bn1
Node layer2_0_relu is from module layer2.0.relu
Node layer2_0_conv2 is from module layer2.0.conv2
Node layer2_0_bn2 is from module layer2.0.bn2
Node layer2_0_downsample_0 is from module layer2.0.downsample.0
Node layer2_0_downsample_1 is from module layer2.0.downsample.1
Node add_2 is from module layer2.0
Node layer2_0_relu_1 is from module layer2.0.relu
Node layer2_1_conv1 is from module layer2.1.conv1
Node layer2_1_bn1 is from module layer2.1.bn1
Node layer2_1_relu is from module layer2.1.relu
Node layer2_1_conv2 is from module layer2.1.conv2
Node layer2_1_bn2 is from module layer2.1.bn2
Node add_3 is from module layer2.1
Node layer2_1_relu_1 is from module layer2.1.relu
Node layer3_0_conv1 is from module layer3.0.conv1
Node layer3_0_bn1 is from module layer3.0.bn1
Node layer3_0_relu is from module layer3.0.relu
Node layer3_0_conv2 is from module layer3.0.conv2
Node layer3_0_bn2 is from module layer3.0.bn2
Node layer3_0_downsample_0 is from module layer3.0.downsample.0
Node layer3_0_downsample_1 is from module layer3.0.downsample.1
Node add_4 is from module layer3.0
Node layer3_0_relu_1 is from module layer3.0.relu
Node layer3_1_conv1 is from module layer3.1.conv1
Node layer3_1_bn1 is from module layer3.1.bn1
Node layer3_1_relu is from module layer3.1.relu
Node layer3_1_conv2 is from module layer3.1.conv2
Node layer3_1_bn2 is from module layer3.1.bn2
Node add_5 is from module layer3.1
Node layer3_1_relu_1 is from module layer3.1.relu
Node layer4_0_conv1 is from module layer4.0.conv1
Node layer4_0_bn1 is from module layer4.0.bn1
Node layer4_0_relu is from module layer4.0.relu
Node layer4_0_conv2 is from module layer4.0.conv2
Node layer4_0_bn2 is from module layer4.0.bn2
Node layer4_0_downsample_0 is from module layer4.0.downsample.0
Node layer4_0_downsample_1 is from module layer4.0.downsample.1
Node add_6 is from module layer4.0
Node layer4_0_relu_1 is from module layer4.0.relu
Node layer4_1_conv1 is from module layer4.1.conv1
Node layer4_1_bn1 is from module layer4.1.bn1
Node layer4_1_relu is from module layer4.1.relu
Node layer4_1_conv2 is from module layer4.1.conv2
Node layer4_1_bn2 is from module layer4.1.bn2
Node add_7 is from module layer4.1
Node layer4_1_relu_1 is from module layer4.1.relu
Node avgpool is from module avgpool
Node flatten is from module 
Node fc is from module fc
Node output is from module None
"""
