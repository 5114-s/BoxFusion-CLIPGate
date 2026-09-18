#!/usr/bin/env python3
"""Fail-closed audit for a paired M1 / M1+M2-nativelogit run."""

import argparse
import json
import os
import pickle

import numpy as np


def _load(path):
    with open(path, "rb") as handle:
        return pickle.load(handle)


def _same_row_geometry(left, right):
    return int(left[0]) == int(right[0]) and np.array_equal(
        np.asarray(left[1]), np.asarray(right[1])
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--scenes", required=True)
    parser.add_argument("--base-root", required=True)
    parser.add_argument("--m1-root", required=True)
    parser.add_argument("--m2-root", required=True)
    parser.add_argument("--out", required=True)
    args = parser.parse_args()

    with open(args.scenes, encoding="utf-8") as handle:
        scenes = [
            line.strip() for line in handle
            if line.strip() and not line.lstrip().startswith("#")
        ]

    totals = {
        "scenes": len(scenes),
        "native_rows": 0,
        "birth_rows": 0,
        "m2_changed_native_rows": 0,
        "m2_decreased_native_rows": 0,
    }
    for scene in scenes:
        base = _load(os.path.join(args.base_root, f"{scene}_boxes.pkl"))
        m1 = _load(os.path.join(args.m1_root, f"{scene}_boxes.pkl"))
        m2 = _load(os.path.join(args.m2_root, f"{scene}_boxes.pkl"))
        if len(base) != len(m1) or len(m1) != len(m2):
            raise AssertionError(f"{scene}: payload snapshot count differs")

        native_count = len(base[0])
        if len(m1[0]) != len(m2[0]) or len(m1[0]) < native_count:
            raise AssertionError(f"{scene}: row count/order contract failed")
        totals["native_rows"] += native_count
        totals["birth_rows"] += len(m1[0]) - native_count

        for index, (m1_row, m2_row) in enumerate(zip(m1[0], m2[0])):
            if not _same_row_geometry(m1_row, m2_row):
                raise AssertionError(f"{scene}: M1/M2 geometry differs at row {index}")
            m1_score = float(m1_row[2])
            m2_score = float(m2_row[2])
            if index < native_count:
                base_row = base[0][index]
                if not _same_row_geometry(base_row, m1_row):
                    raise AssertionError(f"{scene}: native geometry differs at row {index}")
                if not np.isclose(float(base_row[2]), m1_score, atol=0.0, rtol=0.0):
                    raise AssertionError(f"{scene}: M1 changed native score at row {index}")
                if m2_score < m1_score - 1e-12:
                    totals["m2_decreased_native_rows"] += 1
                if not np.isclose(m1_score, m2_score, atol=1e-12, rtol=0.0):
                    totals["m2_changed_native_rows"] += 1
            elif not np.isclose(m1_score, m2_score, atol=0.0, rtol=0.0):
                raise AssertionError(f"{scene}: M2 changed birth score at row {index}")

        for snapshot_index, (m1_snapshot, m2_snapshot) in enumerate(
            zip(m1[1:], m2[1:]), start=1
        ):
            if len(m1_snapshot) != len(m2_snapshot):
                raise AssertionError(
                    f"{scene}: tail row count differs at snapshot {snapshot_index}"
                )
            for row_index, (m1_row, m2_row) in enumerate(
                zip(m1_snapshot, m2_snapshot)
            ):
                if not _same_row_geometry(m1_row, m2_row) or not np.isclose(
                    float(m1_row[2]), float(m2_row[2]), atol=0.0, rtol=0.0
                ):
                    raise AssertionError(
                        f"{scene}: tail differs at snapshot {snapshot_index}, row {row_index}"
                    )

    if totals["m2_decreased_native_rows"]:
        raise AssertionError("M2-nativelogit decreased one or more native scores")
    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as handle:
        json.dump(totals, handle, indent=2, sort_keys=True)
        handle.write("\n")
    print(json.dumps(totals, sort_keys=True))


if __name__ == "__main__":
    main()
