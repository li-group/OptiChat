#!/usr/bin/env python3
"""
Create a smaller-scenario SMPS folder by keeping only the first N scenarios
from an existing .sto file and renormalizing scenario probabilities.

Example:
    python make_smps_subset.py --src MPTSPs_D0_50 --dst MPTSPs_D0_50_S10 --n 10
"""
from __future__ import annotations

import argparse
import shutil
from pathlib import Path


def find_one(folder: Path, suffix: str) -> Path:
    matches = sorted(folder.glob(f"*{suffix}"))
    if len(matches) != 1:
        raise RuntimeError(f"Expected exactly one {suffix} file in {folder}, found {len(matches)}: {matches}")
    return matches[0]


def rename_problem_line(line: str, keyword: str, new_name: str) -> str:
    # MPS/SMPS first lines look like: NAME          problem_name
    if line.lstrip().startswith(keyword):
        return f"{keyword:<14}{new_name}\n"
    return line


def update_first_line(path: Path, keyword: str, new_name: str) -> None:
    lines = path.read_text().splitlines(keepends=True)
    if lines:
        lines[0] = rename_problem_line(lines[0], keyword, new_name)
    path.write_text("".join(lines))


def split_sto(lines: list[str]):
    prefix: list[str] = []
    blocks: list[list[str]] = []
    endata = "ENDATA\n"

    i = 0
    while i < len(lines) and not lines[i].lstrip().startswith("SC "):
        prefix.append(lines[i])
        i += 1

    while i < len(lines):
        stripped = lines[i].strip()
        if stripped == "ENDATA":
            endata = lines[i]
            break
        if not lines[i].lstrip().startswith("SC "):
            raise RuntimeError(f"Expected scenario header at STO line {i+1}, got: {lines[i]!r}")
        block = [lines[i]]
        i += 1
        while i < len(lines):
            stripped = lines[i].strip()
            if stripped == "ENDATA" or lines[i].lstrip().startswith("SC "):
                break
            block.append(lines[i])
            i += 1
        blocks.append(block)

    return prefix, blocks, endata


def set_probability_in_header(header: str, probability: float) -> str:
    # Expected header format: SC SCEN001 ROOT 0.01 STAGE2
    tokens = header.split()
    if len(tokens) < 4 or tokens[0] != "SC":
        raise RuntimeError(f"Unexpected scenario header format: {header!r}")
    tokens[3] = f"{probability:.16g}"
    return " " + " ".join(tokens) + "\n"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--src", required=True, help="Source SMPS folder, e.g., MPTSPs_D0_50")
    parser.add_argument("--dst", required=True, help="Destination SMPS folder, e.g., MPTSPs_D0_50_S10")
    parser.add_argument("--n", type=int, required=True, help="Number of scenarios to keep")
    parser.add_argument("--no-renormalize", action="store_true", help="Keep original scenario probabilities")
    args = parser.parse_args()

    src = Path(args.src).resolve()
    dst = Path(args.dst).resolve()
    if not src.is_dir():
        raise RuntimeError(f"Source folder does not exist: {src}")
    if dst.exists():
        raise RuntimeError(f"Destination already exists; remove it first: {dst}")
    if args.n <= 0:
        raise RuntimeError("--n must be positive")

    dst.mkdir(parents=True)
    new_name = dst.name

    src_cor = find_one(src, ".cor")
    src_tim = find_one(src, ".tim")
    src_sto = find_one(src, ".sto")
    src_lp_files = sorted(src.glob("*.lp"))

    dst_cor = dst / f"{new_name}.cor"
    dst_tim = dst / f"{new_name}.tim"
    dst_sto = dst / f"{new_name}.sto"

    shutil.copy2(src_cor, dst_cor)
    shutil.copy2(src_tim, dst_tim)
    update_first_line(dst_cor, "NAME", new_name)
    update_first_line(dst_tim, "TIME", new_name)

    sto_lines = src_sto.read_text().splitlines(keepends=True)
    prefix, blocks, endata = split_sto(sto_lines)
    if args.n > len(blocks):
        raise RuntimeError(f"Requested {args.n} scenarios, but STO contains only {len(blocks)}")

    chosen = blocks[: args.n]
    if not args.no_renormalize:
        p = 1.0 / args.n
        for block in chosen:
            block[0] = set_probability_in_header(block[0], p)

    if prefix:
        prefix[0] = rename_problem_line(prefix[0], "STOCH", new_name)

    dst_sto.write_text("".join(prefix + [line for block in chosen for line in block] + [endata]))

    # LP is not required by mpi-sppy, but copy/rename it if present for inspection.
    for src_lp in src_lp_files:
        dst_lp = dst / f"{new_name}.lp"
        shutil.copy2(src_lp, dst_lp)
        break

    print(f"Wrote smaller SMPS instance to: {dst}")
    print(f"Scenarios kept: {args.n}")
    if not args.no_renormalize:
        print(f"Each scenario probability set to: {1.0 / args.n:.16g}")
    print("Files:")
    for f in sorted(dst.iterdir()):
        print(f"  {f.name}")


if __name__ == "__main__":
    main()
