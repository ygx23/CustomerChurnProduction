#!/bin/bash
set -e

python3 src/data_validation.py
python3 src/eda.py
python3 src/feature_engineering.py
python3 src/train.py
python3 src/explain.py
