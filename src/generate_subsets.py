import argparse
import random
import numpy as np
import json
from dataset import SMCDataset


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--smc_data_dir', default='data/smcalflow_cs/source_domain_with_target_num32')
    parser.add_argument('--seed', default=0, type=int)
    parser.add_argument('-o', '--output_path', default='data/eval_subsets.json')
    args = parser.parse_args()

    print(args)
    print()

    random.seed(args.seed)
    np.random.seed(args.seed)

    dataset = SMCDataset(
        smc_dir=args.smc_data_dir,
        embedding=None,
        shuffle_train=False
    )

    subsets = [f'val100_seed{i}' for i in range(5)] + [f'test100_seed{i}' for i in range(5)]

    res = {}
    for subset_name in subsets:
        if 'val' in subset_name:
            examples = list(dataset.dev_examples())
        elif 'test' in subset_name:
            examples = list(dataset.test_examples())
        ids = np.random.permutation(len(examples))[:100]
        res[subset_name] = [examples[i].datum_id.to_dict() for i in ids]

    with open(args.output_path, 'w') as f:
        json.dump(res, f, indent=2)


if __name__ == "__main__":
    main()
