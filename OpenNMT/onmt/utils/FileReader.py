local FileReader = torch.class("FileReader")

function FileReader:__init(filename)
  self.file = assert(io.open(filename, "r"))
end

--[[ Read next line in the file and split it on spaces. If EOF is reached, returns nil. ]]
function FileReader:next()
  local line = self.file:read()

  if line == nil then
    return nil
  end

  local sent = {}
  for word in line:gmatch'([^%s]+)' do
    table.insert(sent, word)
  end

  return sent
end

function FileReader:close()
  self.file:close()
end

return FileReader
