import os

from icecream import ic


def print_nodes_info():
    Node = os.environ['LOCAL_RANK']
    ic(Node, os.environ['LOCAL_RANK'], os.environ['RANK'], os.environ['WORLD_SIZE'], os.environ['MASTER_ADDR'],
       os.environ['MASTER_PORT'])
    print()
