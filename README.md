# PyTorch Examples

![Run Examples](https://github.com/pytorch/examples/workflows/Run%20Examples/badge.svg)

https://pytorch.org/examples/

`pytorch/examples` is a repository showcasing examples of using [PyTorch](https://github.com/pytorch/pytorch). The goal is to have curated, short, few/no dependencies _high quality_ examples that are substantially different from each other that can be emulated in your existing work.

- For tutorials: https://github.com/pytorch/tutorials
- For changes to pytorch.org: https://github.com/pytorch/pytorch.github.io
- For a general model hub: https://pytorch.org/hub/ or https://huggingface.co/models
- For recipes on how to run PyTorch in production: https://github.com/facebookresearch/recipes
- For general Q&A and support: https://discuss.pytorch.org/

## Available models

- [Image classification (MNIST) using Convnets](./mnist/README.md)
- [Word-level Language Modeling using RNN and Transformer](./word_language_model/README.md)
- [Training Imagenet Classifiers with Popular Networks](./imagenet/README.md)
- [Generative Adversarial Networks (DCGAN)](./dcgan/README.md)
- [Variational Auto-Encoders](./vae/README.md)
- [Superresolution using an efficient sub-pixel convolutional neural network](./super_resolution/README.md)
- [Hogwild training of shared ConvNets across multiple processes on MNIST](mnist_hogwild)
- [Training a CartPole to balance in OpenAI Gym with actor-critic](./reinforcement_learning/README.md)
- [Natural Language Inference (SNLI) with GloVe vectors, LSTMs, and torchtext](snli)
- [Time sequence prediction - use an LSTM to learn Sine waves](./time_sequence_prediction/README.md)
- [Implement the Neural Style Transfer algorithm on images](./fast_neural_style/README.md)
- [Reinforcement Learning with Actor Critic and REINFORCE algorithms on OpenAI gym](./reinforcement_learning/README.md)
- [PyTorch Module Transformations using fx](./fx/README.md)
- Distributed PyTorch examples with [Distributed Data Parallel](./distributed/ddp/README.md) and [RPC](./distributed/rpc)
- [Several examples illustrating the C++ Frontend](cpp)
- [Image Classification Using Forward-Forward](./mnist_forward_forward/README.md)
- [Language Translation using Transformers](./language_translation/README.md)



Additionally, a list of good examples hosted in their own repositories:

- [Neural Machine Translation using sequence-to-sequence RNN with attention (OpenNMT)](https://github.com/OpenNMT/OpenNMT-py)

## Contributing

If you'd like to contribute your own example or fix a bug please make sure to take a look at [CONTRIBUTING.md](CONTRIBUTING.md).
