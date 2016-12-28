import math
import torch.optim as optim

class Optim(object):

    def _makeOptimizer(self):
        if self.method == 'sgd':
            self.optimizer = optim.SGD(self.params, lr=self.lr)
        elif self.method == 'adagrad':
            self.optimizer = optim.adagrad(self.params, lr=self.lr)
        elif self.method == 'adadelta':
            self.optimizer = optim.adadelta(self.params, lr=self.lr)
        elif self.method == 'adam':
            self.optimizer = optim.adam(self.params, lr=self.lr)
        else:
            raise RuntimeError("Invalid optim method: " + self.method)

    def __init__(self, params, method, lr, lr_decay=1, start_decay_at=None):
        self.params = params
        self.last_ppl = None
        self.lr = lr
        self.method = method
        self.lr_decay = lr_decay
        self.start_decay_at = start_decay_at

        self._makeOptimizer()


    def step(self, params, max_grad_norm):
        # Compute gradients norm.
        grad_norm = 0
        for param in params:
            grad_norm = grad_norm + math.pow(param.grad.norm(), 2)

        grad_norm = math.sqrt(grad_norm)
        shrinkage = max_grad_norm / grad_norm

        for param in params:
            if shrinkage < 1:
                param.grad.mul_(shrinkage)

        self.optimizer.step()

    # decay learning rate if val perf does not improve or we hit the start_decay_at limit
    def updateLearningRate(self, ppl, epoch):
        if self.start_decay_at is not None and epoch >= self.start_decay_at:
            self.start_decay = True
        if self.last_ppl is not None and ppl > self.last_ppl:
            self.start_decay = True

        if self.start_decay:
            self.lr = self.lr * self.lr_decay

        self.last_ppl = ppl

        self._makeOptimizer()
