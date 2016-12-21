local function logJsonRecursive(obj)
  if type(obj) == 'string' then
    io.write('"' .. obj .. '"')
  elseif type(obj) == 'table' then
    local first = true

    io.write('{')

    for key, val in pairs(obj) do
      if not first then
        io.write(',')
      else
        first = false
      end
      io.write('"' .. key .. '":')
      logJsonRecursive(val)
    end

    io.write('}')
  else
    io.write(tostring(obj))
  end
end

--[[ Recursively outputs a Lua object to a JSON objects followed by a new line. ]]
local function logJson(obj)
  logJsonRecursive(obj)
  io.write('\n')
end

return {
  logJson = logJson
}
