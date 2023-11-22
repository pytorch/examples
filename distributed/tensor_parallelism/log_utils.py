import logging
import torch

logging.basicConfig(
    format="%(asctime)s %(message)s", datefmt="%m/%d/%Y %I:%M:%S %p", level=logging.INFO
)

def get_logger():
    return logging.getLogger(__name__)


def rank_log(_rank, logger, msg):
    """helper function to log only on global rank 0"""
    if _rank == 0:
        logger.info(f" {msg}")


def verify_min_gpu_count(min_gpus: int = 2) -> bool:
    """ verification that we have at least 2 gpus to run dist examples """
    has_cuda = torch.cuda.is_available()
    gpu_count = torch.cuda.device_count()
    return has_cuda and gpu_count >= min_gpus
