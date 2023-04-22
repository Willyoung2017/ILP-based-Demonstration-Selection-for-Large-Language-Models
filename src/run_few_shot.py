import argparse
import os
import random
import numpy as np
from tqdm import trange
import json
from demo_selection import *
from dataset import SMCDataset, TurnIdEncoder
from converters.registry import get_converter
from utils.llm_backend import batch_few_shot_query, few_shot_query, ENGINE_TO_BACKEND
from utils.helpers import bool_flag
import openai
import multiprocessing as mp
from tqdm import tqdm

DEFAULT_SYSTEM_PROMPT = 'You are a helpful assistant that can answer questions according to the OCR and caption of an image. You will be provided with demonstration example with similar input and output format. You should always predict an answer even if the nessacary context is not present.'

EMB_OPTIONS = [
    'openai'
]

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


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--smc_data_dir', default='data/smcalflow_cs/source_domain_with_target_num32')
    parser.add_argument('--emb', default='openai', choices=EMB_OPTIONS)
    parser.add_argument('-o', '--output_path', default='outputs/raw/raw_dev_pred.json')
    parser.add_argument('--seed', default=0, type=int)
    parser.add_argument('-n', '--n_shot', default=24, type=int, help='number of demonstrations')
    parser.add_argument('-c', '--converter', default='question_only', help='example to code converter')
    parser.add_argument('-s', '--selector', default='fixed_random',
                        choices=['fixed_random', 'l2_topk', 'cos_topk', 'cos_topk_len_con', 'ip_cos_topk'],
                        help='demonstration example selector')
    parser.add_argument('-e', '--engine', default='code-davinci-002')
    parser.add_argument('--max_prompt_tokens', default=100, type=int)
    parser.add_argument('--eval_subset', default='val100_seed0', choices=EVAL_SUBSETS)
    parser.add_argument('--batch_size', default=20, type=int)
    parser.add_argument('--batch_mode', default=True, type=bool_flag, nargs='?', const=True)
    parser.add_argument('--system_prompt', default=DEFAULT_SYSTEM_PROMPT)

    parser.add_argument('-np', '--n_processes', default=30, type=int)

    # constraint opt args
    parser.add_argument("--max_len", default=1000, type=int, help="demo token len constraints")
    parser.add_argument("--pre_comp", action="store_true", help="whether use pre-computed demo ids")
    parser.add_argument("--diverse", action="store_true", help="whether add diversity constraints")

    args = parser.parse_args()

    print(args)
    print()

    random.seed(args.seed)
    np.random.seed(args.seed)

    if not os.path.exists(os.path.dirname(args.output_path)):
        os.makedirs(os.path.dirname(args.output_path))

    converter = get_converter(args.converter)

    is_chatgpt = ENGINE_TO_BACKEND[args.engine] == 'openai-chat'

    dataset = SMCDataset(
        smc_dir=args.smc_data_dir,
        max_prompt_tokens=args.max_prompt_tokens,
        embedding=args.emb
    )
    train_examples = [example for example in dataset.train_examples()]

    if 'val' in args.eval_subset:
        eval_examples = [example for example in dataset.dev_examples()]
    elif 'test' in args.eval_subset:
        eval_examples = [example for example in dataset.test_examples()]
    else:
        raise ValueError(f'Unknown eval subset {args.eval_subset}')

    if args.eval_subset not in ('val', 'test'):
        with open('data/eval_subsets.json') as fin:
            eval_subsets = json.load(fin)
        turn_ids = eval_subsets[args.eval_subset]
        turn_ids = {(d["dialogue_id"], d["turn_index"]) for d in turn_ids}
        eval_examples = [example for example in eval_examples if
                         (example.datum_id.dialogue_id, example.datum_id.turn_index) in turn_ids]

    print('Number of train examples:', len(train_examples))
    print('Number of eval examples:', len(eval_examples))

    selector_class = {
        'fixed_random': FixedRandomDemoSelection,
        'l2_topk': L2TopKDemoSelection,
        'cos_topk': CosineTopKDemoSelection,
        'cos_topk_len_con': CosineTopKLengthConstrainedDemoSelection,
        'ip_cos_topk': IPCosineTopKDemoSelection,
    }[args.selector]

    if "con" not in args.selector:
        selector = selector_class(
            examples=train_examples,
            n_shot=args.n_shot,
            n_processes=args.n_processes,
        )
    else:
        pre_comp_dir = "data/pre_comp_demo"
        os.makedirs(pre_comp_dir, exist_ok=True)
        selector = selector_class(
            examples=train_examples,
            n_shot=args.n_shot,
            length=args.max_len,
            load_from_pre_comp=args.pre_comp,
            pre_comp_dir=pre_comp_dir,
            diverse=args.diverse,
            n_processes=args.n_processes,
        )

    raw_preds = []
    for i in trange(0, len(eval_examples), args.batch_size, disable=True):
        j = min(i + args.batch_size, len(eval_examples))
        batch = eval_examples[i:j]

        batch_demos = selector.batch_get_demo(batch)

        prompts = [
            converter.example2code(demos=demos, target=example)
            for demos, example in zip(batch_demos, batch)
        ]

        # print(prompts[0])

        addition_kwargs = {} if not is_chatgpt else {'system_prompt': args.system_prompt}

        if args.batch_mode:
            responses = batch_few_shot_query(
                prompts=prompts,
                engine=args.engine,
                max_tokens=100,
                stop_token='\n',
                temperature=0.0,
                top_p=1.0,
                **addition_kwargs
            )
        else:
            responses = [few_shot_query(
                prompt=prompt,
                engine=args.engine,
                max_tokens=100,
                stop_token='\n',
                temperature=0.0,
                top_p=1.0,
                **addition_kwargs
            ) for prompt in prompts]

        for example, prompt, completion in zip(batch, prompts, responses):
            pred = converter.code2answer(prompt + completion).strip()
            raw_preds.append(
                {
                    'dialogue_id': example.datum_id,
                    'user_utterance': example.user_utterance,
                    'agent_utterance': example.agent_utterance,
                    "prompt": prompt,
                    'prediction': pred,
                }
            )

    with open(args.output_path, 'w') as fout:
        for pred in raw_preds:
            fout.write(json.dumps(pred, cls=TurnIdEncoder) + "\n")
    print(f'raw outputs saved to {args.output_path}')

    if "con" in args.selector and not args.pre_comp:
        print(f'pre_comp demos saved to data/pre_comp_demo')
        selector.save_demos()


if __name__ == '__main__':
    main()
