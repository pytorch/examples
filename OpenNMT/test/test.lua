require('onmt.init')

local tester = torch.Tester()


local stringTest = torch.TestSuite()

function stringTest.noSplit()
  tester:eq(onmt.utils.String.split('foo-foo', '\\|'), { 'foo-foo' })
end
function stringTest.emptySplit2()
  tester:eq(onmt.utils.String.split('\\|', '\\|'), { '', '' })
end
function stringTest.emptySplit1Right()
  tester:eq(onmt.utils.String.split('foo\\|', '\\|'), { 'foo', '' })
end
function stringTest.emptySplit1Middle()
  tester:eq(onmt.utils.String.split('foo\\|\\|bar', '\\|'), { 'foo', '', 'bar' })
end
function stringTest.emptySplit1Left()
  tester:eq(onmt.utils.String.split('\\|foo', '\\|'), { '', 'foo' })
end
function stringTest.split2()
  tester:eq(onmt.utils.String.split('foo\\|bar', '\\|'), { 'foo', 'bar' })
end
function stringTest.split3()
  tester:eq(onmt.utils.String.split('foo\\|bar\\|foobar', '\\|'), { 'foo', 'bar', 'foobar' })
end
function stringTest.doNotSplitEscaped1()
  tester:eq(onmt.utils.String.split('foo\\\\|bar', '\\|'), { 'foo\\\\|bar' })
end
function stringTest.doNotSplitEscaped2()
  tester:eq(onmt.utils.String.split('foo\\\\|bar\\|foobar', '\\|'), { 'foo\\\\|bar', 'foobar' })
end
function stringTest.doNotSplitEscaped3()
  tester:eq(onmt.utils.String.split('\\\\|', '\\|'), { '\\\\|' })
end

function stringTest.noStrip()
  tester:eq(onmt.utils.String.strip('foo'), 'foo')
end
function stringTest.stripLeft()
  tester:eq(onmt.utils.String.strip('  foo'), 'foo')
end
function stringTest.stripRight()
  tester:eq(onmt.utils.String.strip('foo  '), 'foo')
end
function stringTest.stripBoth()
  tester:eq(onmt.utils.String.strip('    foo  '), 'foo')
end

tester:add(stringTest)


local tensorTest = torch.TestSuite()

function tensorTest.reuseSmaller()
  local a = torch.Tensor(10, 200)
  local b = onmt.utils.Tensor.reuseTensor(a, { 5, 200 })
  tester:eq(torch.pointer(b:storage()), torch.pointer(a:storage()))
end

function tensorTest.reuseSame()
  local a = torch.Tensor(10, 200)
  local b = onmt.utils.Tensor.reuseTensor(a, a:size())
  tester:eq(torch.pointer(b:storage()), torch.pointer(a:storage()))
end

function tensorTest.reuseMultipleResize()
  local a = torch.Tensor(10, 200)
  local b = onmt.utils.Tensor.reuseTensor(a, { 5, 200 })
  tester:eq(torch.pointer(b:storage()), torch.pointer(a:storage()))
  local c = onmt.utils.Tensor.reuseTensor(a, { 10, 200 })
  tester:eq(torch.pointer(c:storage()), torch.pointer(a:storage()))
end

tester:add(tensorTest)


local dictTest = torch.TestSuite()

function dictTest.createEmptyDict()
  local d = onmt.utils.Dict.new()
  tester:eq(d:size(), 0)
end

function dictTest.createSimpleDict()
  local d = onmt.utils.Dict.new()
  d:add('foo')
  d:add('bar')
  d:add('foobar')

  tester:eq(d:size(), 3)

  tester:ne(d:lookup('foo'), nil)
  tester:ne(d:lookup('bar'), nil)
  tester:ne(d:lookup('foobar'), nil)
end

function dictTest.createSimpleDictWithSpecialTokens()
  local d = onmt.utils.Dict.new({'toto', 'titi'})
  d:add('foo')
  d:add('bar')
  d:add('foobar')

  tester:eq(d:size(), 5)
  tester:eq(#d.special, 2)
  tester:ne(d:lookup('toto'), nil)
  tester:ne(d:lookup('titi'), nil)
end

function dictTest.createSimpleDictDuplicate()
  local d = onmt.utils.Dict.new()
  d:add('foo')
  d:add('bar')
  d:add('foobar')
  d:add('bar')

  tester:eq(d:size(), 3)
  tester:eq(d.frequencies[d:lookup('bar')], 2)
end

function dictTest.pruneLarger()
  local d = onmt.utils.Dict.new()
  d:add('foo')
  d:add('bar')
  d:add('foobar')
  d:add('bar')

  local pruned = d:prune(5)
  tester:eq(torch.pointer(pruned), torch.pointer(d))
end

function dictTest.pruneSmaller()
  local d = onmt.utils.Dict.new()
  d:add('foo')
  d:add('bar')
  d:add('foobar')
  d:add('bar')
  d:add('foobar')
  d:add('bar')

  local pruned = d:prune(2)
  tester:eq(pruned:size(), 2)
  tester:eq(pruned:lookup('foo'), nil)
  tester:ne(pruned:lookup('bar'), nil)
  tester:ne(pruned:lookup('foobar'), nil)
end

function dictTest.pruneSmallerWithSpecialTokens()
  local d = onmt.utils.Dict.new({ 'toto', 'titi' })
  d:add('foo')
  d:add('bar')
  d:add('foobar')
  d:add('bar')
  d:add('foobar')
  d:add('bar')

  local pruned = d:prune(2)
  tester:eq(pruned:size(), 2 + 2)
  tester:eq(pruned:lookup('foo'), nil)
  tester:ne(pruned:lookup('bar'), nil)
  tester:ne(pruned:lookup('foobar'), nil)
  tester:ne(pruned:lookup('toto'), nil)
  tester:ne(pruned:lookup('titi'), nil)
end

function dictTest.pruneInvariableSpecialTokensIndex()
  local d = onmt.utils.Dict.new({ 'toto', 'titi' })
  d:add('foo')
  d:add('bar')
  d:add('foobar')
  d:add('bar')
  d:add('foobar')
  d:add('bar')

  local totoIndex = d:lookup('toto')
  local titiIndex = d:lookup('titi')

  local pruned = d:prune(2)
  tester:eq(pruned:lookup('toto'), totoIndex)
  tester:eq(pruned:lookup('titi'), titiIndex)
end

tester:add(dictTest)


local nmttest = torch.TestSuite()

-- local function equal(t1, t2, msg)
--    if (torch.type(t1) == "table") then
--       for k, _ in pairs(t2) do
--          equal(t1[k], t2[k], msg)
--       end
--    else
--       tester:eq(t1, t2, 0.00001, msg)
--    end
-- end


function nmttest.Data()
end

tester:add(nmttest)

function onmt.test(tests, fixedSeed)
  -- Limit number of threads since everything is small
  local nThreads = torch.getnumthreads()
  torch.setnumthreads(1)

   -- Randomize stuff
  local seed = fixedSeed or (1e5 * torch.tic())
  print('Seed: ', seed)
  math.randomseed(seed)
  torch.manualSeed(seed)
  tester:run(tests)
  torch.setnumthreads(nThreads)
  return tester
end
