"""
This file demonstrates using a custom FX Tracer to override
the behavior of `torch.autograd.profiler.record_function` and
make profiler ranges appear in FX-traced code. This is done
with Python dynamic patching magic, allowing us to explicitly
emit calls to
`torch.ops.profiler._record_function_enter/_record_function_exit`.

Please note that before https://github.com/pytorch/pytorch/pull/65180 lands,
these ranges may be elimineated by `Graph.eliminate_dead_code`
"""
import torch
import torch.fx

# Setup: a module with `record_function`
class Foo(torch.nn.Module):
  def forward(self, x):
    with torch.profiler.record_function('foo'):
      return torch.relu(x)

f = Foo()
x = torch.randn(5, 3, 2)
with torch.autograd.profiler.profile() as prof:
  f(x)

print(prof)
# "foo" range is correctly recorded with normal execution
"""
-------------------  ------------  ------------  ------------  ------------  ------------  ------------  
               Name    Self CPU %      Self CPU   CPU total %     CPU total  CPU time avg    # of Calls  
-------------------  ------------  ------------  ------------  ------------  ------------  ------------  
        aten::zeros         6.10%      10.298us        10.04%      16.943us      16.943us             1  
        aten::empty         2.88%       4.857us         2.88%       4.857us       4.857us             1  
        aten::zero_         1.06%       1.788us         1.06%       1.788us       1.788us             1  
                foo        21.28%      35.925us        89.96%     151.888us     151.888us             1  
        aten::empty        11.59%      19.572us        11.59%      19.572us      19.572us             1  
         aten::relu        23.81%      40.203us        57.09%      96.391us      96.391us             1  
    aten::clamp_min         3.87%       6.539us        33.28%      56.188us      56.188us             1  
        aten::empty         1.09%       1.847us         1.09%       1.847us       1.847us             1  
    aten::clamp_min        28.31%      47.802us        28.31%      47.802us      47.802us             1  
-------------------  ------------  ------------  ------------  ------------  ------------  ------------  
Self CPU time total: 168.831us
"""


traced = torch.fx.symbolic_trace(f)
with torch.autograd.profiler.profile() as prof:
  traced(x)

print(prof)
# "foo" range is not recorded with FX tracing
"""
-------------------  ------------  ------------  ------------  ------------  ------------  ------------  
               Name    Self CPU %      Self CPU   CPU total %     CPU total  CPU time avg    # of Calls  
-------------------  ------------  ------------  ------------  ------------  ------------  ------------  
         aten::relu        23.50%      10.618us       100.00%      45.186us      45.186us             1  
    aten::clamp_min        18.05%       8.154us        76.50%      34.568us      34.568us             1  
        aten::empty        11.77%       5.317us        11.77%       5.317us       5.317us             1  
    aten::clamp_min        46.69%      21.097us        46.69%      21.097us      21.097us             1  
-------------------  ------------  ------------  ------------  ------------  ------------  ------------  
Self CPU time total: 45.186us
"""


class ProfilerTracer(torch.fx.Tracer):
  def trace(self, root, concrete_args=None):
    orig_record_function_enter = torch.autograd.profiler.record_function.__enter__
    orig_record_function_exit = torch.autograd.profiler.record_function.__exit__

    def fake_profiler_enter(_self):
      nonlocal self
      handle_proxy = self.create_proxy(
          kind='call_function',
          target=torch.ops.profiler._record_function_enter,
          args=(_self.name,),
          kwargs={})
      
      assert getattr(_self, '_fx_profiler_ctx', None) is None
      setattr(_self, '_fx_profiler_ctx', handle_proxy)
      return handle_proxy

    def fake_profiler_exit(_self, exc_type, exc_value, traceback):
      assert hasattr(_self, '_fx_profiler_ctx')
      handle_proxy = _self._fx_profiler_ctx
      torch.ops.profiler._record_function_exit(handle_proxy)
      setattr(_self, '_fx_profiler_ctx', None)

    torch.autograd.profiler.record_function.__enter__ = fake_profiler_enter
    torch.autograd.profiler.record_function.__exit__ = fake_profiler_exit

    try:
      return super().trace(root, concrete_args)
    finally:
      torch.autograd.profiler.record_function.__enter__ = orig_record_function_enter
      torch.autograd.profiler.record_function.__exit__ = orig_record_function_exit

pt = ProfilerTracer()

graph_with_profiler = pt.trace(f)
traced_with_profiler = torch.fx.GraphModule(pt.root, graph_with_profiler)

with torch.autograd.profiler.profile() as prof:
  traced_with_profiler(x)

print(prof)
# "foo" range is recorded with special tracer behavior
"""
-------------------  ------------  ------------  ------------  ------------  ------------  ------------  
               Name    Self CPU %      Self CPU   CPU total %     CPU total  CPU time avg    # of Calls  
-------------------  ------------  ------------  ------------  ------------  ------------  ------------  
                foo        19.76%      39.928us       100.00%     202.055us     202.055us             1  
        aten::empty         3.93%       7.950us         3.93%       7.950us       7.950us             1  
         aten::relu        33.79%      68.282us        76.30%     154.177us     154.177us             1  
    aten::clamp_min        27.32%      55.198us        42.51%      85.895us      85.895us             1  
        aten::empty         1.28%       2.585us         1.28%       2.585us       2.585us             1  
    aten::clamp_min        13.91%      28.112us        13.91%      28.112us      28.112us             1  
-------------------  ------------  ------------  ------------  ------------  ------------  ------------  
Self CPU time total: 202.055us
"""
