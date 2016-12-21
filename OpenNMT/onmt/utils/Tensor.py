# --[[ Recursively call `clone()` on all tensors within `out`. ]]
# local function recursiveClone(out)
#   if torch.isTensor(out) then
#     return out:clone()
#   else
#     local res = {}
#     for k, v in ipairs(out) do
#       res[k] = recursiveClone(v)
#     end
#     return res
#   end
# end
#
# local function recursiveSet(dst, src)
#   if torch.isTensor(dst) then
#     dst:set(src)
#   else
#     for k, _ in ipairs(dst) do
#       recursiveSet(dst[k], src[k])
#     end
#   end
# end
#
# --[[ Clone any serializable Torch object. ]]
# local function deepClone(obj)
#   local mem = torch.MemoryFile("w"):binary()
#   mem:writeObject(obj)
#
#   local reader = torch.MemoryFile(mem:storage(), "r"):binary()
#   local clone = reader:readObject()
#
#   reader:close()
#   mem:close()
#
#   return clone
# end
#
# --[[
# Reuse Tensor storage and avoid new allocation unless any dimension
# has a larger size.
#
# Parameters:
#
#   * `t` - the tensor to be reused
#   * `sizes` - a table or tensor of new sizes
#
# Returns: a view on zero-tensor `t`.
#
# --]]
# local function reuseTensor(t, sizes)
#   assert(t ~= nil, 'tensor must not be nil for it to be reused')
#
#   if torch.type(sizes) == 'table' then
#     sizes = torch.LongStorage(sizes)
#   end
#
#   -- Tensor was uninitialized, just resize it.
#   if t:dim() == 0 then
#     return t:resize(sizes):zero()
#   end
#
#   assert(#sizes == t:dim(), 'reused tensor must have the same number of dimensions')
#
#   -- Otherwise, prepare new tensor sizes.
#   local newSizes = t:size()
#
#   for d = 1, t:dim() do
#     -- Change size only if storage needs to expand.
#     if sizes[d] > t:size(d) then
#       newSizes[d] = sizes[d]
#     end
#   end
#
#   -- If one dimension size changed, resize the given tensor.
#   if torch.any(torch.ne(torch.Tensor(torch.totable(newSizes)),
#                         torch.Tensor(torch.totable(t:size())))) then
#     t:resize(newSizes)
#   end
#
#   -- In all cases, extract the given sizes.
#   for d = 1, t:dim() do
#     t = t:narrow(d, 1, sizes[d])
#   end
#
#   return t:zero()
# end
#
# --[[
# Reuse all Tensors within the table with new sizes.
#
# Parameters:
#
#   * `tab` - the table of tensors
#   * `sizes` - a table of new sizes
#
# Returns: a table of tensors using the same storage as `tab`.
#
# --]]
# local function reuseTensorTable(tab, sizes)
#   local newTab = {}
#
#   for i = 1, #tab do
#     table.insert(newTab, reuseTensor(tab[i], sizes))
#   end
#
#   return newTab
# end
#
# --[[
# Initialize a table of tensors with the given sizes.
#
# Parameters:
#
#   * `tab` - the table of tensors
#   * `proto` - tensor to be clone for each index
#   * `sizes` - a table of new sizes
#
# Returns: an initialized table of tensors.
#
# --]]
# local function initTensorTable(size, proto, sizes)
#   local tab = {}
#
#   local base = reuseTensor(proto, sizes)
#
#   for _ = 1, size do
#     table.insert(tab, base:clone())
#   end
#
#   return tab
# end
#
# --[[
# Copy tensors from `src` reusing all tensors from `proto`.
#
# Parameters:
#
#   * `proto` - the table of tensors to be reused
#   * `src` - the source table of tensors
#
# Returns: a copy of `src`.
#
# --]]
# local function copyTensorTable(proto, src)
#   local tab = reuseTensorTable(proto, src[1]:size())
#
#   for i = 1, #tab do
#     tab[i]:copy(src[i])
#   end
#
#   return tab
# end
#
# return {
#   recursiveClone = recursiveClone,
#   recursiveSet = recursiveSet,
#   deepClone = deepClone,
#   reuseTensor = reuseTensor,
#   reuseTensorTable = reuseTensorTable,
#   initTensorTable = initTensorTable,
#   copyTensorTable = copyTensorTable
# }
