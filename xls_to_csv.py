#!/usr/bin/env python3
"""Batch convert tab-separated .xls files in ./data to CSV."""

import csv
import glob
import os


def convert_xls_to_csv(xls_path: str, csv_path: str) -> None:
    with open(xls_path, encoding="iso-8859-1", newline="") as infile:
        reader = csv.reader(infile, delimiter="\t")
        with open(csv_path, "w", encoding="utf-8", newline="") as outfile:
            writer = csv.writer(outfile)
            for row in reader:
                writer.writerow(row)


def main() -> None:
    data_dir = os.path.join(os.path.dirname(__file__), "data")
    xls_files = glob.glob(os.path.join(data_dir, "**", "*.xls"), recursive=True)

    if not xls_files:
        print("No .xls files found in ./data")
        return

    for xls_path in sorted(xls_files):
        csv_path = os.path.splitext(xls_path)[0] + ".csv"
        convert_xls_to_csv(xls_path, csv_path)
        print(f"Converted: {os.path.relpath(xls_path)} -> {os.path.relpath(csv_path)}")

    print(f"\nDone. {len(xls_files)} file(s) converted.")


if __name__ == "__main__":
    main()
