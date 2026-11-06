#!/usr/bin/env bash
# Rebuild everything: data -> detection -> valuation -> statistics -> figures -> tables.
set -e
mkdir -p out paper/tables paper/figures
python3 code/01_prepare.py   > out/log_prepare.txt
python3 code/02_detect.py    > out/log_detect.txt
python3 code/03_value.py     > out/log_value.txt
python3 code/04_analysis.py  > out/log_analysis.txt 2>&1
python3 code/05_figures.py   > out/log_figures.txt 2>&1
python3 code/06_tables.py    > out/log_tables.txt
echo done
