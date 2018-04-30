#!/usr/bin/env python
import argparse
import torch
import torch.nn.functional as F

parser = argparse.ArgumentParser()
parser.add_argument('--poly-degree', type=int, default=4, metavar='N', help='dimension of linear model')
parser.add_argument('--batch-size', type=int, default=32, metavar='N', help='input batch size')
parser.add_argument('--max-iteration', type=int, default=5000, metavar='N', help='the max iteration to stop')
parser.add_argument('--learning-rate', type=float, default=0.1, metavar='F', help='learning rate for gradient descent')
parser.add_argument('--acceptable-loss', type=float, default=1e-3, metavar='F', help='stop training when loss below this value')
parser.add_argument('--var', type=float, default=5., metavar='F', help='variance of the weight of target model')
args = parser.parse_args()

# define a fixed target model: y = b + w0 x + w1 x^2 + w2 x + ...
model_target = torch.nn.Linear(args.poly_degree, 1)
model_target.weight = torch.nn.Parameter(torch.randn(1, args.poly_degree) * args.var)
model_target.bias = torch.nn.Parameter(torch.randn(1) * args.var)
model_target.weight.requires_grad = False
model_target.bias.requires_grad = False

# define a model for regression
model = torch.nn.Linear(args.poly_degree, 1)

def poly_desc(linear_model):
	W = linear_model.weight.view(-1) # flatten
	b = linear_model.bias.item()
	result = 'y = '
	for i, w in enumerate(W):
		result += '{:+.2f} x^{} '.format(w, len(W) - i)
	result += '{:+.2f}'.format(b)
	return result

def get_batch(batch_size=32):
	""" Builds features 'x' i.e. a matrix with columns [x, x^2, x^3, x^4]. """
	random = torch.randn(batch_size, 1)
	x = torch.cat([random ** i for i in range(1, args.poly_degree + 1)], dim=1)
	y = model_target(x)
	return x, y

def train(max_iteration):
	for batch_idx in range(1, max_iteration):
		batch_x, batch_y = get_batch(args.batch_size)

		model.zero_grad()
		pred_y = model(batch_x)
		loss = F.smooth_l1_loss(pred_y, batch_y)
		loss.backward() # Back propagation (compute gradient)

		# Apply gradients
		for param in model.parameters():
			param.data.add_(-args.learning_rate * param.grad.data)

		if loss < args.acceptable_loss:
			break

	print('Loss: {:.6f} after {}/{} batches'.format(loss.item(), batch_idx, max_iteration))

if __name__ == '__main__':
	train(args.max_iteration)
	print('==> Learned function: ' + poly_desc(model))
	print('==> Actual  function: ' + poly_desc(model_target))
