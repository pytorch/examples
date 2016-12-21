require('nn')
require('nngraph')

local Cuda = {
  activated = false
}

function Cuda.init(opt, gpuIdx)
  Cuda.activated = opt.gpuid > 0

  if Cuda.activated then
    local _, err = pcall(function()
      require('cutorch')
      require('cunn')
      if gpuIdx == nil then
        -- allow memory access between devices
        cutorch.getKernelPeerToPeerAccess(true)
        if opt.seed then
          cutorch.manualSeedAll(opt.seed)
        end
        cutorch.setDevice(opt.gpuid)
      else
        cutorch.setDevice(gpuIdx)
      end
      if opt.seed then
        cutorch.manualSeed(opt.seed)
      end
    end)

    if err then
      error(err)
    end
  end
end

--[[
  Recursively move all supported objects in `obj` on the GPU.
  When using CPU only, converts to float instead of the default double.
]]
function Cuda.convert(obj)
  if torch.typename(obj) then
    if Cuda.activated and obj.cuda ~= nil then
      return obj:cuda()
    elseif not Cuda.activated and obj.float ~= nil then
      -- Defaults to float instead of double.
      return obj:float()
    end
  end

  if torch.typename(obj) or type(obj) == 'table' then
    for k, v in pairs(obj) do
      obj[k] = Cuda.convert(v)
    end
  end

  return obj
end

function Cuda.getGPUs(ngpu)
  local gpus = {}
  if Cuda.activated then
    if ngpu > cutorch.getDeviceCount() then
      error("not enough available GPU - " .. ngpu .. " requested, " .. cutorch.getDeviceCount() .. " available")
    end
    gpus[1] = Cuda.gpuid
    local i = 1
    while #gpus ~= ngpu do
      if i ~= gpus[1] then
        table.insert(gpus, i)
      end
      i = i + 1
    end
  else
    for _ = 1, ngpu do
      table.insert(gpus, 0)
    end
  end
  return gpus
end

function Cuda.freeMemory()
  if Cuda.activated then
    local freeMemory = cutorch.getMemoryUsage(cutorch.getDevice())
    return freeMemory
  end
  return 0
end

return Cuda
