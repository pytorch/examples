PyTorch Examples
================

This pages lists various PyTorch examples that you can use to learn and
experiment with PyTorch.

.. panels::

    Image Classification using Vision Transformer
    ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
 
    This example shows how to train a `Vision Transformer <https://en.wikipedia.org/wiki/Vision_transformer>`__ 
    from scratch on the `CIFAR10 <https://en.wikipedia.org/wiki/CIFAR-10>`__ database.

    `GO TO EXAMPLE <https://github.com/pytorch/examples/blob/main/vision_transformer>`__ :opticon:`link-external` 

    ---

    Image Classification Using ConvNets
    ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
 
    This example demonstrates how to run image classification
    with `Convolutional Neural Networks ConvNets <https://cs231n.github.io/convolutional-networks/>`__
    on the `MNIST <https://en.wikipedia.org/wiki/MNIST_database>`__ database.

    `GO TO EXAMPLE <https://github.com/pytorch/examples/blob/main/mnist>`__ :opticon:`link-external` 

    ---

    Measuring Similarity using Siamese Network
    ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
 
    This example demonstrates how to measure similarity between two images
    using `Siamese network <https://en.wikipedia.org/wiki/Siamese_neural_network>`__
    on the `MNIST <https://en.wikipedia.org/wiki/MNIST_database>`__ database.

    `GO TO EXAMPLE <https://github.com/pytorch/examples/blob/main/siamese_network>`__ :opticon:`link-external` 

    ---

    Word-level Language Modeling using RNN and Transformer
    ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

    This example demonstrates how to train a multi-layer `recurrent neural
    network (RNN) <https://en.wikipedia.org/wiki/Recurrent_neural_network>`__,
    such as Elman, GRU, or LSTM, or Transformer on a language
    modeling task by using the Wikitext-2 dataset. 

    `GO TO EXAMPLE <https://github.com/pytorch/examples/blob/main/word_language_model>`__ :opticon:`link-external`
    ---

    Training ImageNet Classifiers
    ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

    This example demonstrates how you can train some of the most popular
    model architectures, including `ResNet <https://en.wikipedia.org/wiki/Residual_neural_network>`__, 
    `AlexNet <https://en.wikipedia.org/wiki/AlexNet>`__, and `VGG <https://arxiv.org/pdf/1409.1556.pdf>`__
    on the `ImageNet <https://image-net.org/>`__ dataset.

    `GO TO EXAMPLE <https://github.com/pytorch/examples/blob/main/imagenet>`__ :opticon:`link-external`
    ---

    Generative Adversarial Networks (DCGAN)
    ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

    This example implements the `Unsupervised Representation Learning with Deep Convolutional Generative Adversarial Networks  <https://arxiv.org/abs/1511.06434>`__ paper.

    `GO TO EXAMPLE <https://github.com/pytorch/examples/blob/main/dcgan>`__ :opticon:`link-external`
    ---
     
    Variational Auto-Encoders
    ^^^^^^^^^^^^^^^^^^^^^^^^^

    This example implements the `Auto-Encoding Variational Bayes <https://arxiv.org/abs/1312.6114>`__ paper
    with `ReLUs <https://en.wikipedia.org/wiki/Rectifier_(neural_networks)>`__ and the Adam optimizer.

    `GO TO EXAMPLE <https://github.com/pytorch/examples/blob/main/vae>`__ :opticon:`link-external`
    ---
   
    Super-resolution Using an Efficient Sub-Pixel CNN
    ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

    This example demonstrates how to use the sub-pixel convolution layer
    described in `Real-Time Single Image and Video Super-Resolution Using an Efficient Sub-Pixel Convolutional Neural Network <https://arxiv.org/abs/1609.05158>`__ paper. This example trains a super-resolution
    network on the `BSD300 dataset <https://www2.eecs.berkeley.edu/Research/Projects/CS/vision/bsds/>`__. 

    `GO TO EXAMPLE <https://github.com/pytorch/examples/blob/main/super_resolution>`__ :opticon:`link-external`

    ---
    HOGWILD! Training of Shared ConvNets
    ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

    `HOGWILD! <https://arxiv.org/abs/1106.5730>`__ is a scheme that allows
    Stochastic Gradient Descent (SGD)
    parallelization without memory locking. This example demonstrates how
    to perform HOGWILD! training of shared ConvNets on MNIST.

    `GO TO EXAMPLE <https://github.com/pytorch/examples/blob/main/mnist_hogwild>`__ :opticon:`link-external`

    ---
    Training a CartPole to balance in OpenAI Gym with actor-critic
    ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

    This reinforcement learning tutorial demonstrates how to train a
    CartPole to balance
    in the `OpenAI Gym <https://gym.openai.com/>`__ toolkit by using the
    `Actor-Critic <https://proceedings.neurips.cc/paper/1999/file/6449f44a102fde848669bdd9eb6b76fa-Paper.pdf>`__ method.

    `GO TO EXAMPLE <https://github.com/pytorch/examples/blob/main/reinforcement_learning>`__ :opticon:`link-external`
    ---

    Time Sequence Prediction
    ^^^^^^^^^^^^^^^^^^^^^^^^

    This beginner example demonstrates how to use LSTMCell to
    learn sine wave signals to predict the signal values in the future.

    `GO TO EXAMPLE <https://github.com/pytorch/examples/tree/main/time_sequence_prediction>`__ :opticon:`link-external`

    ---

    Implement the Neural Style Transfer algorithm on images
    ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

    This tutorial demonstrates how you can use PyTorch's implementation
    of the `Neural Style Transfer (NST) <https://en.wikipedia.org/wiki/Neural_style_transfer>`__
    algorithm on images.

    `GO TO EXAMPLE <https://github.com/pytorch/examples/blob/main/fast_neural_style>`__ :opticon:`link-external`
    ---

    PyTorch Module Transformations using fx
    ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

    This set of examples demonstrates the torch.fx toolkit. For more
    information about `torch.fx`, see
    `torch.fx Overview <https://pytorch.org/docs/master/fx.html>`__.

    `GO TO EXAMPLE <https://github.com/pytorch/examples/blob/main/fx>`__ :opticon:`link-external`
    ---

    Distributed PyTorch
    ^^^^^^^^^^^^^^^^^^^

    This set of examples demonstrates `Distributed Data Parallel (DDP) <https://pytorch.org/tutorials/intermediate/ddp_tutorial.html>`__ and `Distributed RPC framework <https://pytorch.org/docs/stable/rpc.html>`__. 
    Includes the code used in the `DDP tutorial series <https://pytorch.org/tutorials/beginner/ddp_series_intro.html>`__.

    `GO TO EXAMPLES <https://github.com/pytorch/examples/tree/main/distributed>`__ :opticon:`link-external`
    
    ---

    C++ Frontend
    ^^^^^^^^^^^^

    The PyTorch C++ frontend is a C++14 library for CPU and GPU tensor computation.
    This set of examples includes a linear regression, autograd, image recognition
    (MNIST), and other useful examples using PyTorch C++ frontend.

    `GO TO EXAMPLES <https://github.com/pytorch/examples/tree/main/cpp>`__ :opticon:`link-external`

    ---

    Image Classification Using Forward-Forward Algorithm
    ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

    This example implements the paper `The Forward-Forward Algorithm: Some Preliminary Investigations <https://arxiv.org/pdf/2212.13345.pdf>`__ by Geoffrey Hinton.
    on the `MNIST <https://en.wikipedia.org/wiki/MNIST_database>`__ database.
    It is an introductory example to the Forward-Forward algorithm.

    `GO TO EXAMPLE <https://github.com/pytorch/examples/tree/main/mnist_forward_forward>`__ :opticon:`link-external` 

    ---

    Graph Convolutional Network
    ^^^^^^^^^^^^^^^^^^^^^^^^^^^

    This example implements the `Semi-Supervised Classification with Graph Convolutional Networks <https://arxiv.org/pdf/1609.02907.pdf>`__ paper on the CORA database.

    `GO TO EXAMPLE <https://github.com/pytorch/examples/blob/main/gcn>`__ :opticon:`link-external` 