#!/usr/bin/env python

import json
import argparse
from enum import Enum
from typing import Dict, Any, List


class ProfilerKeys(Enum):
    MemoryPerSec = "dram__bytes.sum.per_second"
    GPUTime = "gpu__time_duration.sum"
    FAddAvg = "smsp__sass_thread_inst_executed_op_fadd_pred_on.avg"
    FAddMax = "smsp__sass_thread_inst_executed_op_fadd_pred_on.max"
    FAddMin = "smsp__sass_thread_inst_executed_op_fadd_pred_on.min"
    FAddSum = "smsp__sass_thread_inst_executed_op_fadd_pred_on.sum"
    FFmaAvg = "smsp__sass_thread_inst_executed_op_ffma_pred_on.avg"
    FFmaMax = "smsp__sass_thread_inst_executed_op_ffma_pred_on.max"
    FFmaMin = "smsp__sass_thread_inst_executed_op_ffma_pred_on.min"
    FFmaSum = "smsp__sass_thread_inst_executed_op_ffma_pred_on.sum"
    FMulAvg = "smsp__sass_thread_inst_executed_op_fmul_pred_on.avg"
    FMulMax = "smsp__sass_thread_inst_executed_op_fmul_pred_on.max"
    FMulMin = "smsp__sass_thread_inst_executed_op_fmul_pred_on.min"
    FMulSum = "smsp__sass_thread_inst_executed_op_fmul_pred_on.sum"


def get_total_flops(tables: List[dict]) -> int:
    def get_kernel_flops(table: Dict[str, Any]) -> int:
        return int(
            table[ProfilerKeys.FAddSum.value]["val"]
            + table[ProfilerKeys.FMulSum.value]["val"]
            + 2 * table[ProfilerKeys.FFmaSum.value]["val"]
        )

    return sum(get_kernel_flops(table) for table in tables)


def get_total_memory(tables: List[dict]) -> float:
    def get_memory(table: Dict[str, Any]) -> float:
        memory_metrics = table[ProfilerKeys.MemoryPerSec.value]
        m_unit, m_val = memory_metrics["unit"], memory_metrics["val"]

        # Convert to GBytes/second
        if m_unit == 'Mbyte/second':
            m_val /= 10**3
        elif m_unit == "Tbyte/second":
            m_val *= 10**3
        elif m_unit != 'Gbyte/second':
            raise Exception(f"Invalid Memory unit: {m_unit}")

        time_metrics = table[ProfilerKeys.GPUTime.value]
        t_unit, t_val = time_metrics["unit"], time_metrics["val"]

        # Convert to seconds
        if t_unit == "usecond":
            t_val /= 10 ** 6
        elif t_unit == "msecond":
            t_val /= 10 ** 3
        elif t_unit != "second":
            raise Exception(f"Invalid time unit: {t_unit}")

        return m_val * t_val

    return sum(get_memory(table) for table in tables)


def parse_args():
    parser = argparse.ArgumentParser(
        prog="NCU Stats", description="Compute stats from NCU tables json file."
    )
    parser.add_argument("file", help="NCU table json file path.", type=str)
    parser.add_argument(
        "--stat", help="Type of computation to do.", default=[], nargs="+"
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    with open(args.file, "r") as f:
        tables = json.load(f)

    if 'ai' in args.stat or not args.stat:
        args.stat = ["flops", "memory", "ai"]

    if "flops" in args.stat:
        flops = get_total_flops(tables)
        print(f"Total FLOPs: {(flops / 10**9):.2f} GFLOPs")
    if "memory" in args.stat:
        memory = get_total_memory(tables)
        print(f"Total Memory: {memory:.2f} GB")

    if "ai" in args.stat:
        print(f"AI: {(flops / (memory * 10**9)):.2f} FLOPs/bytes")
