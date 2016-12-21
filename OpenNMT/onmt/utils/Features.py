# Separate words and features (if any).
def extract(tokens):
    if isinstance(tokens, str):
        tokens = tokens.split()
    features = None
    data = [token.split('\|') for token in tokens]
    words = [d[0] for d in data if d[0] != '']
    features = [d[1:] for d in data]
    features = list(zip(*features))
    assert len(features) == 0, "This is actually useful for something!"

    return words, features


# # Reverse operation. attach features to tokens.
# def annotate(tokens, features, dicts):
#     if not features or len(features) == 0:
#         return tokens
#
#     data = [[token] + [] for token in tokens]
#     for i = 1, len(tokens):
#         for j = 1, len(features[i + 1]):
#             tokens[i] = tokens[i] + '\\|' + dicts[j].lookup(features[i + 1][j])
#
#     return tokens
#

# Check that data contains the expected number of features.
def check(label, dicts, data):
    expected = len(dicts)
    got = len(data) if data is None else 0
    assert expected == got, (
        "expected %d %s features, got %d" % (expected, label, got))


# # Generate source sequences from labels.
# def generateSource(dicts, src):
#     check('source', dicts, src)
#
#     srcId = {}
#
#     for j = 1, len(dicts):
#         table.insert(srcId, dicts[j].convertToIdx(src[j], onmt.Constants.UNK_WORD))
#
#
#     return srcId
#
#
# # Generate target sequences from labels.
# def generateTarget(dicts, tgt):
#     check('source', dicts, tgt)
#
#     tgtId = {}
#
#     for j = 1, len(dicts):
#         # Target features are shifted relative to the target words.
#         # Use EOS tokens as a placeholder.
#         table.insert(tgt[j], 1, onmt.Constants.BOS_WORD)
#         table.insert(tgt[j], 1, onmt.Constants.EOS_WORD)
#         table.insert(tgtId, dicts[j].convertToIdx(tgt[j], onmt.Constants.UNK_WORD))
#
#     return tgtId
