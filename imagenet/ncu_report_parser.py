#!/usr/bin/env python

import argparse
import json
import re
from typing import Tuple, List, Dict


def parse_table_line(line: str) -> Tuple[str, str, float]:
    tokens = line.split()
    if len(tokens) != 3:
        raise Exception(f"Invalid table entry: {line}")
    tokens[2] = tokens[2].replace(",", "")
    return tokens[0], tokens[1], float(tokens[2])


def extract_table_entries(lines: List[str]) -> List[Dict[str, dict]]:
    parse_table = False
    line_no = 0
    tables = []
    table = {}
    while line_no < len(lines):
        line = lines[line_no]
        if not parse_table and line == "    Section: Command line profiler metrics\n":
            line_no += 1  # Skip one line separator
            parse_table = True
            table = {}
        elif parse_table:
            if re.match(r"^[\-\s]+$", line):
                parse_table = False
                tables.append(table)
            else:
                key, unit, val = parse_table_line(line)
                table[key] = {"unit": unit, "val": val}

        line_no += 1

    return tables


def parse_args():
    parser = argparse.ArgumentParser(
        prog="NCU Report Parser", description="Parse profiling data into json."
    )
    parser.add_argument("file", help="NCU report file path.", type=str)
    return parser.parse_args()


if __name__ == "__main__":
    ncu_report_path = parse_args().file
    with open(ncu_report_path, "r") as f:
        lines = f.readlines()

    tables = extract_table_entries(lines)

    with open(f"{ncu_report_path.split('.')[0]}_tables.json", "w") as f:
        json.dump(tables, f)
