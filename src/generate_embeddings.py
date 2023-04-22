import openai
import json
import os
import argparse
import numpy as np
from typing import List
from tqdm import tqdm, trange
from sentence_transformers import SentenceTransformer
from dataset import SMCDataset


def batch_get_openai_embedding(text: List[str], model="text-embedding-ada-002"):
    resp = openai.Embedding.create(input=[s.replace("\n", " ") for s in text],
                                   model=model)
    res = [None] * len(text)
    for dic in resp['data']:
        res[dic['index']] = dic['embedding']
    return res


def batch_get_sbert_embedding(text: List[str], model):
    return model.encode(text)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--smc_data_dir', default='data/smcalflow_cs/source_domain_with_target_num32')
    parser.add_argument('--model', type=str, default="text-embedding-ada-002",
                        choices=['all-mpnet-base-v2', 'text-embedding-ada-002'])
    parser.add_argument('--name', default='caption_ocr_question')
    parser.add_argument('--batch_size', type=int, default=100)
    parser.add_argument('--output_dir', default='data/embeddings')
    args = parser.parse_args()
    print(args)

    dataset = SMCDataset(
        smc_dir=args.smc_data_dir,
        embedding=None,
        shuffle_train=False
    )
    examples = list(dataset.train_examples()) + list(dataset.dev_examples()) \
               + list(dataset.test_examples())

    if args.model.startswith('text-embedding-'):
        model_type = 'openai'
    else:
        model_type = 'sbert'
        sbert_model = SentenceTransformer(args.model, device='cuda')

    all_utterance= [ex.user_utterance for ex in examples]
    all_plans = [ex.agent_utterance for ex in examples]

    model_dim = {
        'all-mpnet-base-v2': 768,
        'text-embedding-ada-002': 1536
    }[args.model]

    for all_text, name in [(all_utterance, 'utterance'), (all_plans, 'plan')]:
        X = np.zeros((len(examples), model_dim), dtype=np.float32)
        for i in trange(0, len(all_text), args.batch_size):
            j = min(i + args.batch_size, len(all_text))
            if model_type == 'openai':
                embeddings = batch_get_openai_embedding(all_text[i:j], model=args.model)
            elif model_type == 'sbert':
                embeddings = sbert_model.encode(all_text[i:j])
            else:
                raise ValueError('Unknown model type')
            X[i:j] = np.array(embeddings, dtype=np.float32)

        output_path = os.path.join(args.output_dir, f'{model_type}_{name}.npy')
        np.save(output_path, X)
        print(f'Saved to {output_path}')


if __name__ == '__main__':
    main()
