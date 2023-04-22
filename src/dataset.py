import random
from dataclasses import dataclass
import os
import collections
import gzip
from json import JSONEncoder

import numpy as np
import json
from typing import List, Optional, Iterator


class TurnIdEncoder(JSONEncoder):
    def default(self, obj):
        if isinstance(obj, TurnId):
            return str(obj)
        return super().default(obj)


@dataclass(frozen=True)
class TurnId:
    dialogue_id: str
    turn_index: int

    def __hash__(self):
        return hash((self.dialogue_id, self.turn_index))

    def __str__(self):
        return f"{self.dialogue_id}_{self.turn_index}"


@dataclass
class SMCExample:
    datum_id: TurnId
    # the current user utterance
    user_utterance: str
    # the next agent utterance
    agent_utterance: str
    utterance_emb: Optional[np.ndarray] = None
    plan_emb: Optional[np.ndarray] = None


class SMCDataset:
    def __init__(
            self,
            smc_dir='data/smcalflow_cs/source_domain_with_target_num32',
            max_prompt_tokens: Optional[int] = None,
            embedding=None,
            shuffle_train=True,
    ):
        self.train = self._load_questions(os.path.join(
            smc_dir, 'train.jsonl'))
        self.dev = self._load_questions(os.path.join(
            smc_dir, 'valid.jsonl'))
        self.test = self._load_questions(os.path.join(
            smc_dir, 'test.jsonl'))

        self.max_prompt_tokens = max_prompt_tokens

        if embedding is not None:
            self._load_embeddings(emb_name=embedding)

        if shuffle_train:
            random.shuffle(self.train)

    @staticmethod
    def _load_questions(path) -> List[SMCExample]:
        res = []
        with open(path, 'r') as fin:
            for line in fin:
                dic = json.loads(line)
                res.append(SMCExample(
                    datum_id=TurnId(dialogue_id=dic['dialogue_id'], turn_index=dic['turn_index']),
                    user_utterance=dic['utterance'],
                    agent_utterance=dic["plan"],
                ))
        return res

    def _load_embeddings(self, emb_name) -> None:
        utterance_emb = np.load(f'data/embeddings/{emb_name}_utterance.npy')
        plan_emb = np.load(f'data/embeddings/{emb_name}_plan.npy')
        examples = list(self.train) + list(self.dev) + list(self.test)
        assert len(examples) == utterance_emb.shape[0] == plan_emb.shape[0]
        for i, ex in enumerate(examples):
            ex.utterance_emb = utterance_emb[i]
            ex.plan_emb = plan_emb[i]

    def train_examples(self) -> Iterator[SMCExample]:
        for example in self.train:
            yield example

    def dev_examples(self) -> Iterator[SMCExample]:
        for example in self.dev:
            yield example

    def test_examples(self) -> Iterator[SMCExample]:
        for example in self.test:
            yield example
