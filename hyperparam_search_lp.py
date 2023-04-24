import os
EVAL_SUBSETS = [
    'val',
    'test',
    'val100_seed0',
    'val100_seed1',
    'val100_seed2',
    'val100_seed3',
    'val100_seed4',
    'test100_seed0',
    'test100_seed1',
    'test100_seed2',
    'test100_seed3',
    'test100_seed4',
]

if __name__ == "__main__":
    for eval_subset in EVAL_SUBSETS[2:]:
        for n_shot in 2, 3, 5, 8, 12, 24:
            for max_len in 500, 1000:
                for selector, np in zip(("cos_topk_len_con", "cos_topk_len_con_greedy"), (32, 1)):
                    out_fn = f"outputs/raw/{eval_subset}_n{n_shot}_l{max_len}_{selector}.json"
                    cmd = f"python src/run_few_shot.py -s {selector} -np {np} -n {n_shot} --max_len {max_len} -o {out_fn}"
                    os.system(cmd)
