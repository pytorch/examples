
--[[Parse and lookup a words from a phrase table.
--]]
local PhraseTable = torch.class('PhraseTable')


function PhraseTable:__init(filePath)
  local f = assert(io.open(filePath, 'r'))

  self.table = {}

  for line in f:lines() do
    local c = line:split("|||")
    self.table[onmt.utils.String.strip(c[1])] = c[2]
  end

  f:close()
end

--[[ Return the phrase table match for `word`. ]]
function PhraseTable:lookup(word)
  return self.table[word]
end

function PhraseTable:contains(word)
  return self:lookup(word) ~= nil
end

return PhraseTable
