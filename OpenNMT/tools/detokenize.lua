local unicode = require './utils/unicode'

local cmd = torch.CmdLine()

local sep_marker = '\\@'
local feat_marker = '\\|'
local protect_char = '\\'

cmd:text("")
cmd:text("**detokenize.lua**")
cmd:text("")

cmd:option('-case_feature', false, [[First feature is case feature]])

local opt = cmd:parse(arg)

local function analyseToken(t)
  local feats = {}
  local tok = ""
  local p
  local leftsep = false
  local rightsep = false
  local i = 1
  while i <= #t do
    if t:sub(i, i) == protect_char and t:sub(i+1, i+1) == protect_char then
      i = i + 1
    elseif t:sub(i, i+#feat_marker-1) == feat_marker then
      tok = t:sub(1, i-1)
      p = i
      break
    end
    tok = tok .. t:sub(i, i)
    i = i + 1
  end
  if tok:sub(1,#sep_marker) == sep_marker then
    tok = tok:sub(1+#sep_marker)
    leftsep = true
  end
  if tok:sub(-#sep_marker,-1) == sep_marker then
    tok = tok:sub(1,-#sep_marker-1)
    rightsep = true
  end
  if p then
    p = p + #feat_marker
    local j = p
    while j <= #t do
      if t[j] == protect_char and t[j+1] == protect_char then
        j = j + 1
      elseif t:sub(j, j+#feat_marker-1) == feat_marker then
        table.insert(feats, t:sub(p, j-1))
        j = j + #feat_marker - 1
        p = j + 1
      end
      j = j + 1
    end
    table.insert(feats, t:sub(p))
  end
  return tok, leftsep, rightsep, feats
end

local function getTokens(t)
  local fields = {}
  t:gsub("([^ ]+)", function(tok)
    local w, leftsep, rightsep, feats =  analyseToken(tok)
    table.insert(fields, { w=w, leftsep=leftsep, rightsep=rightsep, feats=feats })
  end)
  return fields
end

local function restoreCase(w, feats)
  if opt.case_feature then
    assert(#feats>=1)
    if feats[1] == 'L' or feats[1] == 'N' then
      return w
    else
      local wr = ''
      for v, c in unicode.utf8_iter(w) do
        if wr == '' or feats[1] == 'U' then
          local _, cu = unicode.getUpper(v)
          if cu then c = cu end
        end
        wr = wr .. c
      end
      return wr
    end
  end
  return w
end

-- minimalistic tokenization
-- - remove utf-8 BOM character
-- - turn sequences of separators into single space
-- - skip any other non control character [U+0001-U+002F]
-- - keep sequence of letters/numbers and tokenize everything else

local function detokenize(line)
  local dline = ""
  local tokens = getTokens(line)
  for j = 1, #tokens do
    if j > 1 and not tokens[j-1].rightsep and not tokens[j].leftsep then
      dline = dline .. " "
    end
    dline = dline .. restoreCase(tokens[j].w, tokens[j].feats)
  end
  return dline
end

local timer = torch.Timer()
local idx = 1
for line in io.lines() do
  local res, err = pcall(function() io.write(detokenize(line) .. '\n') end)
  if not res then
    if string.find(err,"interrupted") then
      error("interrupted")
    else
      error("unicode error in line "..idx..": "..line..'-'..err)
    end
  end
  idx = idx + 1
end

io.stderr:write(string.format('Detokenization completed in %0.3f seconds - %d sentences\n',timer:time().real,idx-1))
