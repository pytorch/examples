--[[ Append table `src` to `dst`. ]]
local function append(dst, src)
  for i = 1, #src do
    table.insert(dst, src[i])
  end
end

--[[ Reorder table `tab` based on the `index` array. ]]
local function reorder(tab, index)
  local newTab = {}
  for i = 1, #tab do
    table.insert(newTab, tab[index[i]])
  end
  return newTab
end
