#!/usr/bin/env python
"""
Logistic regression example

Trains a single fully-connected layer to learn a quadratic function.
"""
from __future__ import print_function

import torch
import torch.autograd
import torch.nn


WEIGHTS = torch.randn(2, 1) * 5
BIAS = torch.randn(1) * 5


def get_features(xs):
    return torch.FloatTensor([[x * x, x] for x in xs])


def get_targets(features):
    return features.mm(WEIGHTS) + BIAS[0]


def print_function(weights, bias):
    print('y = (%f * x^2) + (%f * x) + %f' % (
        weights[0], weights[1], bias[0],
    ))


def get_batch(batch_size=32):
    xs = torch.randn(batch_size)
    features = get_features(xs)
    targets = get_targets(features)
    return (
        torch.autograd.Variable(features),
        torch.autograd.Variable(targets),
    )


if __name__ == '__main__':

    # Model definition
    fc = torch.nn.Linear(WEIGHTS.size()[0], 1)
    l1 = torch.nn.L1Loss()

    batches = 0
    while True:
        batches += 1

        # Get data
        batch_x, batch_y = get_batch()

        # Reset gradients
        for param in fc.parameters():
            param.grad.zero_()

        # Forward pass
        output = l1(fc(batch_x), batch_y)
        loss = output.data[0]

        # Backward pass
        output.backward()

        # Apply gradients
        for param in fc.parameters():
            param.data.add_(-1 * param.grad)

        # Stop criterion
        if loss < 0.1:
            break

    print('Loss: %f after %d batches' % (loss, batches))

    print('==> Learned function')
    print_function(fc.weight.data.view(-1), fc.bias.data)

    print('==> Actual function')
    print_function(WEIGHTS.view(-1), BIAS)
