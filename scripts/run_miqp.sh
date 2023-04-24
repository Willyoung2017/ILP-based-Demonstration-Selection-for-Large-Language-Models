for n in 10 24; do
  for seed in {0..4}; do
    python src/run_few_shot.py -np 1 --n_shot $n -s cos_topk --whiten --eval_subset "val100_seed$seed"
    python src/evaluate.py --predictions_json outputs/raw/raw_dev_pred.json
  done
done

for n in 10 24; do
  for alpha in 0.001 0.003 0.01 0.03 0.1 0.3 1; do
    for seed in {0..4}; do
      python src/run_few_shot.py -np 1 --n_shot $n -s miqp --whiten --top_sim 50 --div_alpha $alpha --eval_subset "val100_seed$seed"
      python src/evaluate.py --predictions_json outputs/raw/raw_dev_pred.json
    done
  done
done
