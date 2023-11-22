

def rank_log(_rank, logger, msg):
    """helper function to log only on global rank 0"""
    if _rank == 0:
        logger.info(f" {msg}")
