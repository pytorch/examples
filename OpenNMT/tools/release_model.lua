require('onmt.init')

local path = require('pl.path')

local cmd = torch.CmdLine()
cmd:option('-model', '', 'trained model file')
cmd:option('-output_model', '', 'released model file')
cmd:option('-gpuid', 0, 'which gpuid to use')
cmd:option('-force', false, 'force output model creation')
local opt = cmd:parse(arg)

local function main()
  assert(path.exists(opt.model), 'model \'' .. opt.model .. '\' does not exist.')

  if opt.output_model:len() == 0 then
    if opt.model:sub(-3) == '.t7' then
      opt.output_model = opt.model:sub(1, -4) -- copy input model without '.t7' extension
    else
      opt.output_model = opt.model
    end
    opt.output_model = opt.output_model .. '_release.t7'
  end

  if not opt.force then
    assert(not path.exists(opt.output_model),
           'output model already exists; use -force to overwrite.')
  end

  if opt.gpuid > 0 then
    require('cutorch')
    cutorch.setDevice(opt.gpuid)
  end

  print('Loading model \'' .. opt.model .. '\'...')
  local checkpoint = torch.load(opt.model)
  print('... done.')

  print('Converting model...')
  for _, model in pairs(checkpoint.models) do
    for _, net in pairs(model.modules) do
      net:float()
      net:clearState()
    end
  end
  print('... done.')

  print('Releasing model \'' .. opt.output_model .. '\'...')
  torch.save(opt.output_model, checkpoint)
  print('... done.')
end

main()
