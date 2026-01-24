set -e

DATASET="${DATASET:-real_ds}"
for kind in 'общее' 'танцы' 'парное' 'одиночное' 'синхронное'; do
  KIND="$kind" deepeval test run app/bench/test_base.py -id "${DATASET} :: ${kind}"
done
