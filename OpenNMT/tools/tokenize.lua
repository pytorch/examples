local unicode = require './utils/unicode'

local cmd = torch.CmdLine()

local sep_marker = '\\@'
local feat_marker = '\\|'
local protect_char = '\\'

cmd:text("")
cmd:text("**tokenize.lua**")
cmd:text("")

cmd:option('-mode', 'conservative', [[Define how aggressive should the tokenization be - 'aggressive' only keeps sequences of letters/numbers,
                                    'conservative' allows mix of alphanumeric as in: '2,000', 'E65', 'soft-landing']])
cmd:option('-sep_annotate', 'marker', [[Include separator annotation using sep_marker (marker), or feature (feature), or nothing (none)]])
cmd:option('-case_feature', false, [[Generate case feature]])

local opt = cmd:parse(arg)

local function combineCase(feat, case)
  if feat == 'N' then
    if case == 'lower' then feat = 'L' end
    if case == 'upper' then feat = 'C1' end
  elseif feat == 'L' then
    if case == 'upper' then feat = 'M' end
  elseif feat == 'C1' then
    if case == 'upper' then feat = 'U' end
    if case == 'lower' then feat = 'C' end
  elseif feat == 'C' then
    if case == 'upper' then feat = 'M' end
  end
  return feat
end

local function appendMarker(l)
  if opt.case_feature then
    local p=l:find(feat_marker, -4)
    return l:sub(1,p-1)..sep_marker..l:sub(p)
  end
  return l..sep_marker
end

-- minimalistic tokenization
-- - remove utf-8 BOM character
-- - turn sequences of separators into single space
-- - skip any other non control character [U+0001-U+002F]
-- - keep sequence of letters/numbers and tokenize everything else

local function tokenize(line)
  local nline = ""
  local spacefeat = 'N'
  local casefeat = 'N'
  local space = true
  local letter = false
  local number = false
  local other = false
  for v, c, nextv in unicode.utf8_iter(line) do
    if unicode.isSeparator(v) then
      if space == false then
        if opt.sep_annotate=='feature' then nline = nline..feat_marker..spacefeat end
        if opt.case_feature then nline = nline..feat_marker..string.sub(casefeat,1,1) end
        nline = nline..' '
      end
      number = false
      letter = false
      space = true
      other = false
      spacefeat = 'S'
      casefeat = 'N'
    else
      if v > 32 and not(v == 0xFEFF) then
        if c == protect_char then c = protect_char..c end
        local is_letter, case = unicode.isLetter(v)
        if is_letter and opt.case_feature then
          local lu, lc = unicode.getLower(v)
          if lu then c = lc end
        end
        local is_number = unicode.isNumber(v)
        if opt.mode == 'conservative' then
          if is_number or (c == '-' and letter == true) or c == '_' or
             (letter == true and (c == '.' or c == ',') and (unicode.isNumber(nextv) or unicode.isLetter(nextv))) then
            is_letter = true
            case = "other"
          end
        end
        if is_letter then
          if not(letter == true or space == true) then
            if opt.sep_annotate == 'marker' then
              nline = appendMarker(nline)
            end
            if opt.sep_annotate=='feature' then nline = nline..feat_marker..spacefeat end
            if opt.case_feature then nline = nline..feat_marker..string.sub(casefeat,1,1) end
            nline = nline..' '
            spacefeat = 'N'
            casefeat = 'N'
          elseif other == true then
            if opt.sep_annotate == 'marker' then
              nline = appendMarker(nline)
            end
          end
          casefeat = combineCase(casefeat, case)
          nline = nline..c
          space = false
          number = false
          other = false
          letter = true
        elseif is_number then
          if not(number == true or space == true) then
            if opt.sep_annotate == 'marker' then
              if not(letter) then
                nline = appendMarker(nline)
              else
                c = sep_marker..c
              end
            end
            if opt.sep_annotate=='feature' then nline = nline..feat_marker..spacefeat end
            if opt.case_feature then nline = nline..feat_marker..string.sub(casefeat,1,1) end
            nline = nline..' '
            spacefeat = 'N'
            casefeat = 'N'
          elseif other == true then
            if opt.sep_annotate == 'marker' then
              nline = appendMarker(nline)
            end
          end
          nline = nline..c
          space = false
          letter = false
          other = false
          number = true
        else
          if not space == true then
            if opt.sep_annotate == 'marker' then
              c = sep_marker..c
            end
            if opt.sep_annotate=='feature' then nline = nline..feat_marker..spacefeat end
            if opt.case_feature then nline = nline..feat_marker..string.sub(casefeat,1,1) end
            nline = nline .. ' '
            spacefeat = 'N'
            casefeat = 'N'
          elseif other == true then
            if opt.sep_annotate == 'marker' then
              c = sep_marker..c
            end
          end
          nline = nline..c
          if opt.sep_annotate=='feature' then nline = nline..feat_marker..spacefeat end
          if opt.case_feature then nline = nline..feat_marker..string.sub(casefeat,1,1) end
          nline = nline..' '
          number = false
          letter = false
          other = true
          space = true
        end
      end
    end
  end

  -- remove final space
  if space == true then
    nline = string.sub(nline, 1, -2)
  else
    if opt.sep_feature then nline = nline..feat_marker..spacefeat end
    if opt.case_feature then nline = nline..feat_marker..string.sub(casefeat,1,1) end
  end

  return nline
end

local timer = torch.Timer()
local idx = 1
for line in io.lines() do
  local res, err = pcall(function() io.write(tokenize(line) .. '\n') end)
  if not res then
    if string.find(err,"interrupted") then
      error("interrupted")
    else
      error("unicode error in line "..idx..": "..line..'-'..err)
    end
  end
  idx = idx + 1
end

io.stderr:write(string.format('Tokenization completed in %0.3f seconds - %d sentences\n',timer:time().real,idx-1))
