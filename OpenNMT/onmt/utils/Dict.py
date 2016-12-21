class Dict(object):
    def __init__(self, data):
        self.idxToLabel = {}
        self.labelToIdx = {}
        self.frequencies = {}

        # Special entries will not be pruned.
        self.special = {}

        if data != None:
            if type(data) == "string" then # File to load.
                self.loadFile(data)
            else:
                self.addSpecials(data)




#" Return the number of entries in the dictionary. "
def Dict.size():
    return len(self.idxToLabel)


#" Load entries from a file. "
def Dict.loadFile(filename):
    reader = onmt.utils.FileReader(filename)

    while True:
        fields = reader.next()

        if not fields:
            break


        label = fields[1]
        idx = tonumber(fields[2])

        self.add(label, idx)


    reader.close()


#" Write entries to a file. "
def Dict.writeFile(filename):
    file = assert(io.open(filename, 'w'))

    for i = 1, self.size():
        label = self.idxToLabel[i]
        file.write(label + ' ' + i + '\n')


    file.close()


#" Lookup `key` in the dictionary. it can be an index or a string. "
def Dict.lookup(key):
    if type(key) == "string":
        return self.labelToIdx[key]
    else:
        return self.idxToLabel[key]



#" Mark this `label` and `idx` as special (i.e. will not be pruned). "
def Dict.addSpecial(label, idx):
    idx = self.add(label, idx)
    table.insert(self.special, idx)


#" Mark all labels in `labels` as specials (i.e. will not be pruned). "
def Dict.addSpecials(labels):
    for i = 1, len(labels):
        self.addSpecial(labels[i])



#" Add `label` in the dictionary. Use `idx` as its index if given. "
def Dict.add(label, idx):
    if idx != None:
        self.idxToLabel[idx] = label
        self.labelToIdx[label] = idx
    else:
        idx = self.labelToIdx[label]
        if idx == None:
            idx = len(self.idxToLabel) + 1
            self.idxToLabel[idx] = label
            self.labelToIdx[label] = idx



    if self.frequencies[idx] == None:
        self.frequencies[idx] = 1
    else:
        self.frequencies[idx] = self.frequencies[idx] + 1


    return idx


#" Return a new dictionary with the `size` most frequent entries. "
def Dict.prune(size):
    if size >= self.size():
        return self


    # Only keep the `size` most frequent entries.
    freq = torch.Tensor(self.frequencies)
    _, idx = torch.sort(freq, 1, True)

    newDict = Dict.new()

    # Add special entries in all cases.
    for i = 1, len(self.special):
        newDict.addSpecial(self.idxToLabel[self.special[i]])


    for i = 1, size:
        newDict.add(self.idxToLabel[idx[i]])


    return newDict


#[[
    Convert `labels` to indices. Use `unkWord` if not found.
    Optionally insert `bosWord` at the beginning and `eosWord` at the .
]]
def Dict.convertToIdx(labels, unkWord, bosWord, eosWord):
    vec = {}

    if bosWord != None:
        table.insert(vec, self.lookup(bosWord))


    for i = 1, len(labels):
        idx = self.lookup(labels[i])
        if idx == None:
            idx = self.lookup(unkWord)

        table.insert(vec, idx)


    if eosWord != None:
        table.insert(vec, self.lookup(eosWord))


    return torch.IntTensor(vec)


#" Convert `idx` to labels. If index `stop` is reached, convert it and return. "
def Dict.convertToLabels(idx, stop):
    labels = {}

    for i = 1, len(idx):
        table.insert(labels, self.lookup(idx[i]))
        if idx[i] == stop:
            break



    return labels


return Dict
