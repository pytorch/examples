# Copyright (c) 2022 Meta Platforms, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the Apache-style license found in the
# LICENSE file in the root directory of this source tree.

# This is a simple check to confirm that your current server has full bfloat support -
#  both GPU native support, and Network communication support.

# Be warned that if you run on V100 without a check like this, you will be running without native Bfloat16
# support and will find significant performance degradation (but it will not complain via an error).
# Hence the reason for a checker!

from pkg_resources import packaging
import torch
import torch.cuda.nccl as nccl
import torch.distributed as dist

# global flag that confirms ampere architecture, cuda version and
# nccl version to verify bfloat16 native support is ready

def bfloat_support():
    return (
        torch.version.cuda
        and torch.cuda.is_bf16_supported()
        and packaging.version.parse(torch.version.cuda).release >= (11, 0)
        and dist.is_nccl_available()
        and nccl.version() >= (2, 10)
    )
