import torch
import torch.distributed.rpc as rpc
import threading
import os


def call_method(method, rref, *args, **kwargs):
    return method(rref.local_value(), *args, **kwargs)


def _remote_method(method, rref, *args, **kwargs):
    args = [method, rref] + list(args)
    return rpc_sync(rref.owner(), _call_method, args=args, kwargs=kwargs)


class ParameterServer:
    def __init__(self):
        self.params = torch.rand(100)
        self.running = False

    def get_params(self):
        return self.params

    def update_params(self, new_params):
        self.params = new_params
