import argparse
import threading
import concurrent.futures as futures

import torch.distributed.rpc as rpc
from torch.distributed.rpc import rpc_sync, rpc_async, remote

_server = None

def _run_on_server(name, *args, **kwargs):
    return _server.call(name, *args, **kwargs)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Batch RPC example",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    parser.add_argument('--name')
    parser.add_argument('--rank', type=int)
    parser.add_argument('--world_size', type=int)
    parser.add_argument('--server_name')
    return parser.parse_args()


class BatchServer:
    def __init__(self, batch_size=2):
        self._batch_size = batch_size
        self._fns = {}
        self._inps = {}
        self._outs = {}
        self.lock = threading.Lock()
        global _server
        _server = self

    def bind(self, fn):
        name = fn.__name__
        self._fns[name] = fn
        self._inps[name] = []
        self._outs[name] = futures.Future()

    def call(self, fn_name, *args, **kwargs):
        with self.lock:
            inps = self._inps[fn_name]
            fut = self._outs[fn_name]
            idx = len(inps)
            inps.append((args, kwargs))
            if idx + 1 >= self._batch_size:
                self._inps[fn_name] = []
                self._outs[fn_name] = futures.Future()

        if idx + 1 >= self._batch_size:
            rets = []
            for arg, kwargs in inps:
                rets.append(self._fns[fn_name](*arg, **kwargs))
            fut.set_result(rets)

        return fut.result()[idx]


class BatchClient:
    def __init__(self, server_name):
        self._server_info = rpc.get_worker_info(worker_name=server_name)

    def __getattr__(self, name):
        def fn(*args, **kwargs):
            return rpc_sync(
                self._server_info,
                _run_on_server,
                args=(name, *args),
                kwargs=kwargs
            )
        return fn
