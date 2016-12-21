import nn
import nngraph

Cuda = {
    activated = False
}

def Cuda.init(opt, gpuIdx):
    Cuda.activated = opt.gpuid > 0

    if Cuda.activated:
        _, err = pcall(function()
            import cutorch
            import cunn
            if gpuIdx == None:
                # allow memory access between devices
                cutorch.getKernelPeerToPeerAccess(True)
                if opt.seed:
                    cutorch.manualSeedAll(opt.seed)
                
                cutorch.setDevice(opt.gpuid)
            else:
                cutorch.setDevice(gpuIdx)
            
            if opt.seed:
                cutorch.manualSeed(opt.seed)
            
        )

        if err:
            error(err)
        
    


#[[
    Recursively move all supported objects in `obj` on the GPU.
    When using CPU only, converts to float instead of the default double.
]]
def Cuda.convert(obj):
    if torch.typename(obj):
        if Cuda.activated and obj.cuda != None:
            return obj.cuda()
        elif not Cuda.activated and obj.float != None:
            # Defaults to float instead of double.
            return obj.float()
        
    

    if torch.typename(obj) or type(obj) == 'table':
        for k, v in pairs(obj):
            obj[k] = Cuda.convert(v)
        
    

    return obj


def Cuda.getGPUs(ngpu):
    gpus = {}
    if Cuda.activated:
        if ngpu > cutorch.getDeviceCount():
            error("not enough available GPU - " + ngpu + " requested, " + cutorch.getDeviceCount() + " available")
        
        gpus[1] = Cuda.gpuid
        i = 1
        while len(gpus) != ngpu:
            if i != gpus[1]:
                table.insert(gpus, i)
            
            i = i + 1
        
    else:
        for _ = 1, ngpu:
            table.insert(gpus, 0)
        
    
    return gpus


def Cuda.freeMemory():
    if Cuda.activated:
        freeMemory = cutorch.getMemoryUsage(cutorch.getDevice())
        return freeMemory
    
    return 0


return Cuda
