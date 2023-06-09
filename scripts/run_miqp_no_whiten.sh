for n in 5 24; do
  for seed in {0..4}; do
    python -u src/run_few_shot.py -np 1 --n_shot $n -s cos_topk --eval_subset "val100_seed$seed"
    python -u src/evaluate.py --predictions_json outputs/raw/raw_dev_pred.json
  done
  for alpha in 0.001 0.003 0.01 0.03 0.1 0.3 1; do
    for seed in {0..4}; do
      python -u src/run_few_shot.py -np 1 --n_shot $n -s miqp --top_sim 50 --div_alpha $alpha --eval_subset "val100_seed$seed"
      python -u src/evaluate.py --predictions_json outputs/raw/raw_dev_pred.json
    done
  done
done
