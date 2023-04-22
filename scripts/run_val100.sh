for i in {0..2}; do
  python src/run_few_shot.py --eval_subset "val100_seed$i"
  python src/evaluate.py --predictions_json outputs/raw/raw_dev_pred.json
done

for i in {0..2}; do
  python src/run_few_shot.py --eval_subset "val100_seed$i" -s cos_topk
  python src/evaluate.py --predictions_json outputs/raw/raw_dev_pred.json
done
