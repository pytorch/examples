#[[ Feature decoder generator. Given RNN state, produce categorical distribution over
tokens and features.

    Implements $$[softmax(W^1 h + b^1), softmax(W^2 h + b^2), ..., softmax(W^n h + b^n)] $$.
#]]


FeaturesGenerator, parent = torch.class('onmt.FeaturesGenerator', 'nn.Container')

#[[
Parameters.

    * `rnnSize` - Input rnn size.
    * `outputSize` - Output size (number of tokens).
    * `features` - table of feature sizes.
#]]
def FeaturesGenerator.__init(rnnSize, outputSize, features):
    parent.__init(self)
    self.net = self._buildGenerator(rnnSize, outputSize, features)
    self.add(self.net)


def FeaturesGenerator._buildGenerator(rnnSize, outputSize, features):
    generator = nn.ConcatTable()

    # Add default generator.
    generator.add(nn.Sequential()
                    .add(nn.Linear(rnnSize, outputSize))
                    .add(nn.LogSoftMax())
                    .add(nn.SelectTable(1)))

    # Add a generator for each target feature.
    for i = 1, len(features):
        generator.add(nn.Sequential()
                        .add(nn.Linear(rnnSize, features[i]:size()))
                        .add(nn.LogSoftMax()))

    return generator
