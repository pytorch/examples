import torch
import time
import torchvision.models as models
from torch.profiler import profile, record_function, ProfilerActivity, schedule
def trace_handler(p):
    output = p.key_averages().table(sort_by="self_cuda_time_total", row_limit=5)
    prof.export_chrome_trace("trace3.json")
    # print(output)

model = models.resnet18().cuda()
inputs = torch.randn(5, 3, 224, 224).cuda()
inputs2 = torch.randn(256, 3, 224, 224).cuda()
model.eval()
my_schedule = schedule(
    skip_first=2,
    wait=2,
    warmup=2,
    active=10)

with profile(activities=[
        ProfilerActivity.CPU, ProfilerActivity.CUDA], record_shapes=True, schedule=my_schedule, on_trace_ready=trace_handler) as prof:
    for _ in range(8):
        with record_function("inference 1"):
            model(inputs2)
        with record_function("inference 2"):
            model(inputs2)
        prof.step()

# event_list_avg = prof.key_averages()
# imp_events = [(e.cuda_time, e.cpu_time) for e in event_list_avg if e.key in ['inference 1', 'inference 2']]
# print(imp_events)
# print(event_list_avg.table(sort_by="cuda_time_total", row_limit=15))
# torch.cuda.synchronize()
# start_event = torch.cuda.Event(enable_timing=True)
# end_event = torch.cuda.Event(enable_timing=True)

# t_start = time.time()*1000
# start_event.record()
# model(inputs2)
# end_event.record()
# torch.cuda.synchronize()
# t_end = time.time()*1000
# print("Model Execution Time: ",start_event.elapsed_time(end_event))
# print("Wall clock time: ", (t_end-t_start))




