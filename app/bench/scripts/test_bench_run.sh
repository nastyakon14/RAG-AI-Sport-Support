set -e

DATASET="${DATASET:-real_ds}"
deepeval test run app/bench/test_base.py -id "${DATASET}"

