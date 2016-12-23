import torch.multiprocessing as multiprocessing

def _loop(qIn, qOut):
    state = {}
    while True: # FIXME: shutdown
        idx, closure, args = qIn.get()
        qOut.put(closure(idx, args, state))

class ThreadPool(object):
    def __init__(self, nthreads):
        self.nthreads = nthreads
        self.count = max(nthreads, 1)
        if nthreads > 0:
            self.queues  = [(multiprocessing.Queue(), multiprocessing.Queue()) for i in range(nthreads)]
            self.processes = [multiprocessing.Process(
                    target=_loop, args=(qIn,qOut)).start() for qIn, qOut in self.queues]
        else:
            self.state = {'idx': 0}

    def launch(self, label, closure, args=None, endcallback=None):
        if label is not None:
            print("START", label)

        for j in range(self.count):
            if self.nthreads == 0:
                endcallback(closure(args, sef.state))
            else:
                self.queues[j][0].put((j, closure, args))

        for j in range(self.nthreads):
            res = self.queues[j][1].get()
            if endcallback is not None:
                endcallback(res)

        if label is not None:
            print("DONE", label)
#
#
# Parallel = {
#     gpus = {0},
#     _pool = None,
#     count = 1,
#     gradBuffer = torch.Tensor()
# }
#
# # Synchronizes the current stream on dst device with src device. This is only
# # necessary if we are not on the default stream
# def waitForDevice(dst, src):
#       stream = cutorch.getStream()
#       if stream != 0:
#             cutorch.streamWaitForMultiDevice(dst, stream, { [src] = {stream} })
#
#
#
# def Parallel.init(opt):
#     if onmt.utils.Cuda.activated:
#         Parallel.count = opt.nparallel
#         Parallel.gpus = onmt.utils.Cuda.getGPUs(opt.nparallel)
#         Parallel.gradBuffer = onmt.utils.Cuda.convert(Parallel.gradBuffer)
#         if Parallel.count > 1:
#             print('Using ' + Parallel.count + ' threads on ' + len(Parallel.gpus) + ' GPUs')
#             threads = import threads
#             threads.Threads.serialization('threads.sharedserialize')
#             thegpus = Parallel.gpus
#             Parallel._pool = threads.Threads(
#                 Parallel.count,
#                 function(threadid)
#                     import cunn
#                     import nngraph
#                     import onmt.init
#                     onmt.utils.Cuda.init(opt, thegpus[threadid])
#
#             ) # dedicate threads to GPUs
#             Parallel._pool.specific(True)
#
#         if Parallel.count > 1 and not(opt.no_nccl):
#             # check if we have nccl installed
#             ret
#             ret, Parallel.usenccl = pcall(require, 'nccl')
#             if not ret :
#                 print("WARNING. for improved efficiency in nparallel mode - do install nccl")
#                 Parallel.usenccl = None
#             elif os.getenv('CUDA_LAUNCH_BLOCKING') == '1':
#                 print("WARNING. CUDA_LAUNCH_BLOCKING set - cannot use nccl")
#                 Parallel.usenccl = None
#
#
#
#
#
# def Parallel.getGPU(i):
#     if onmt.utils.Cuda.activated and Parallel.gpus[i] != 0:
#         return Parallel.gpus[i]
#
#     return 0
#
#
# #" Launch def in parallel on different threads. ":
# def Parallel.launch(label, closure, callback):
#     callback = endcallback or function() end
#     if label != None:
#         print("START",label)
#
#     for j = 1, Parallel.count:
#         if Parallel._pool == None:
#             callback(closure(j))
#         else:
#             Parallel._pool.addjob(j, function() return closure(j) , endcallback)
#
#
#     if Parallel._pool:
#         Parallel._pool.synchronize()
#
#     if label != None:
#         print("DONE",label)
#
#
#
# #" Accumulate the gradient parameters from the different parallel threads. "
# def Parallel.accGradParams(gradParams, batches):
#     if Parallel.count > 1:
#         for h = 1, len(gradParams[1]):
#             inputs = { gradParams[1][h] }
#             for j = 2, len(batches):
#                 if not Parallel.usenccl:
#                     # TODO - this is memory costly since we need to clone full parameters from one GPU to another
#                     # to avoid out-of-memory, we can copy/add by batch
#
#                   # Synchronize before and after copy to ensure that it doesn't overlap
#                   # with this add or previous adds
#                     waitForDevice(Parallel.gpus[j], Parallel.gpus[1])
#                     remoteGrads = onmt.utils.Tensor.reuseTensor(Parallel.gradBuffer, gradParams[j][h].size())
#                     remoteGrads.copy(gradParams[j][h])
#                     waitForDevice(Parallel.gpus[1], Parallel.gpus[j])
#                     gradParams[1][h].add(remoteGrads)
#                 else:
#                     table.insert(inputs, gradParams[j][h])
#
#
#             if Parallel.usenccl:
#                 Parallel.usenccl.reduce(inputs, None, True)
#
#
#
#
#
# #" Sync parameters from main model to different parallel threads. "
# def Parallel.syncParams(params):
#     if Parallel.count > 1:
#         if not Parallel.usenccl:
#             for j = 2, Parallel.count:
#                 for h = 1, len(params[1]):
#                     params[j][h].copy(params[1][h])
#
#                 waitForDevice(Parallel.gpus[j], Parallel.gpus[1])
#
#         else:
#             for h = 1, len(params[1]):
#                 inputs = { params[1][h] }
#                 for j = 2, Parallel.count:
#                     table.insert(inputs, params[j][h])
#
#                 Parallel.usenccl.bcast(inputs, True, 1)
