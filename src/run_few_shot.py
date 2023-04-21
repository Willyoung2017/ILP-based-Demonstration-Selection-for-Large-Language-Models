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

DEFAULT_SYSTEM_PROMPT = 'You are a helpful assistant that can answer questions according to the OCR and caption of an image. You will be provided with demonstration example with similar input and output format. You should always predict an answer even if the nessacary context is not present.'

# DEFAULT_SYSTEM_PROMPT = 'You are a helpful assistant.'

EMB_OPTIONS = [
    'openai_question',
    'openai_caption',
    'openai_ocr',
    'openai_caption_question',
    'openai_caption_ocr_question',
    'sbert_question',
    'sbert_caption',
    'sbert_ocr',
    'sbert_caption_question',
    'sbert_caption_ocr_question'
]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--smc_data_dir', default='data/smcalflow_cs/source_domain_with_target_num0')
    parser.add_argument('--emb', default=None, choices=EMB_OPTIONS)
    parser.add_argument('-o', '--output_path', default='outputs/raw/raw_dev_pred.json')
    parser.add_argument('--seed', default=0, type=int)
    parser.add_argument('-n', '--n_shot', default=5, type=int, help='number of demonstrations')
    parser.add_argument('-c', '--converter', default='python_question_only', help='example to code converter')
    parser.add_argument('-s', '--selector', default='fixed_random',
                        choices=['fixed_random', 'l2_topk', 'cos_topk'],
                        help='demonstration example selector')
    parser.add_argument('-e', '--engine', default='code-davinci-002')
    parser.add_argument('--max_prompt_tokens', default=100, type=int)
    parser.add_argument('--max_eval_num', default=10, type=int)
    parser.add_argument('--batch_size', default=20, type=int)
    parser.add_argument('--batch_mode', default=True, type=bool_flag, nargs='?', const=True)
    parser.add_argument('--system_prompt', default=DEFAULT_SYSTEM_PROMPT)
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
    dev_examples = [example for example in dataset.dev_examples()]

    if args.max_eval_num > 0:
        dev_examples = dev_examples[:args.max_eval_num]

    selector_class = {
        'fixed_random': FixedRandomDemoSelection,
        'l2_topk': L2TopKDemoSelection,
        'cos_topk': CosineTopKDemoSelection,
    }[args.selector]

    selector = selector_class(
        examples=train_examples,
        n_shot=args.n_shot,
    )

    raw_preds = []
    for i in trange(0, len(dev_examples), args.batch_size):
        j = min(i + args.batch_size, len(dev_examples))
        batch = dev_examples[i:j]

        prompts = [
            converter.example2code(demos=selector.get_demo(example), target=example)
            for example in batch
        ]

        print(prompts[0])

        addition_kwargs = {} if not is_chatgpt else {'system_prompt': args.system_prompt}

        if args.batch_mode:
            responses = batch_few_shot_query(
                prompts=prompts,
                engine=args.engine,
                max_tokens=32,
                stop_token='\n',
                temperature=0.0,
                top_p=1.0,
                **addition_kwargs
            )
        else:
            responses = [few_shot_query(
                prompt=prompt,
                engine=args.engine,
                max_tokens=32,
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
        json.dump(raw_preds, fout, indent=2, cls=TurnIdEncoder)
    print(f'raw outputs saved to {args.output_path}')


if __name__ == '__main__':
    main()