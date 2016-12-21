--[[
  Split `str` on string or pattern separator `sep`.
  Compared to the standard Lua split function, this one does not drop empty fragment
  and do not split if `sep` is escaped with \.
]]
local function split(str, sep)
  local res = {}
  local fragmentIndex = 1
  local searchOffset = 1

  while searchOffset <= str:len() do
    local sepStart, sepEnd = str:find(sep, searchOffset)

    local sub
    if not sepStart then
      sub = str:sub(fragmentIndex)
      table.insert(res, sub)
      fragmentIndex = str:len() + 1
      searchOffset = fragmentIndex
    elseif sepStart > 1 and str:sub(sepStart - 1, sepStart - 1) == '\\' then
      searchOffset = sepStart + 1
    else
      sub = str:sub(fragmentIndex, sepStart - 1)
      table.insert(res, sub)
      fragmentIndex = sepEnd + 1
      searchOffset = fragmentIndex
      if fragmentIndex > str:len() then
        table.insert(res, '')
      end
    end
  end

  return res
end

--[[ Remove whitespaces at the start and end of the string `s`. ]]
local function strip(s)
  return s:gsub("^%s+",""):gsub("%s+$","")
end

--[[ Convenience function to test `s` for emptiness. ]]
local function isEmpty(s)
  return s == nil or s == ''
end

return {
  split = split,
  strip = strip,
  isEmpty = isEmpty
}
