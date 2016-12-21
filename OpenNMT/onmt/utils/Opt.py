# local function isSet(opt, name)
#   return opt[name]:len() > 0
# end
#
# --[[ Check that option `name` is set in `opt`. Throw an error if not set. ]]
# local function requireOption(opt, name)
#   if not isSet(opt, name) then
#     error("option -" .. name .. " is required")
#   end
# end
#
# --[[ Make sure all options in `names` are set in `opt`. ]]
# local function requireOptions(opt, names)
#   for i = 1, #names do
#     requireOption(opt, names[i])
#   end
# end
#
# --[[ Convert `val` string to its actual type (boolean, number or string). ]]
# local function convert(val)
#   if val == 'true' then
#     return true
#   elseif val == 'false' then
#     return false
#   else
#     return tonumber(val) or val
#   end
# end
#
# --[[ Return options set in the file `filename`. ]]
# local function loadFile(filename)
#   local file = assert(io.open(filename, "r"))
#   local opt = {}
#
#   for line in file:lines() do
#     -- Ignore empty or commented out lines.
#     if line:len() > 0 and string.sub(line, 1, 1) ~= '#' then
#       local field = line:split('=')
#       assert(#field == 2, 'badly formatted config file')
#       local key = onmt.utils.String.strip(field[1])
#       local val = onmt.utils.String.strip(field[2])
#       opt[key] = convert(val)
#     end
#   end
#
#   file:close()
#   return opt
# end
#
# --[[ Override `opt` with option values set in file `filename`. ]]
# local function loadConfig(filename, opt)
#   local config = loadFile(filename)
#
#   for key, val in pairs(config) do
#     assert(opt[key] ~= nil, 'unkown option ' .. key)
#     opt[key] = val
#   end
#
#   return opt
# end
#
# local function dump(opt, filename)
#   local file = assert(io.open(filename, 'w'))
#
#   for key, val in pairs(opt) do
#     file:write(key .. ' = ' .. tostring(val) .. '\n')
#   end
#
#   file:close()
# end
#
# local function init(opt, requiredOptions)
#   if opt.config:len() > 0 then
#     opt = loadConfig(opt.config, opt)
#   end
#
#   requireOptions(opt, requiredOptions)
#
#   if opt.seed then
#     torch.manualSeed(opt.seed)
#   end
# end
#
# return {
#   dump = dump,
#   init = init
# }

# --[[ Convert `val` string to its actual type (boolean, number or string). ]]
# local function convert(val)
#   if val == 'true' then
#     return true
#   elseif val == 'false' then
#     return false
#   else
#     return tonumber(val) or val
#   end
# end


def convert(v):
    if v in ('true','True'):
        return True
    if v in ('false','False'):
        return False
    try:
        return int(v)
    except ValueError:
        pass
    try:
        return float(v)
    except ValueError:
        pass
    return v


def loadFile(filename):
    with open(filename) as file:
        opt = {}
        for line in file.readlines():
            field = line.split('=')
            assert len(field) == 2, 'badly formatted config file'
            opt[field[0].strip()] = convert(field[1].strip())


def initConfig(opt):
    if opt.config is not None:
        config = loadFile(opt.config)
        opt.update(config)
