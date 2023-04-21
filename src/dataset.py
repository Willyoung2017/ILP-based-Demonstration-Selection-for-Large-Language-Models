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
    embedding: Optional[np.ndarray] = None



class SMCDataset:
    def __init__(
            self,
            smc_dir='data/smcalflow_cs/source_domain_with_target_num0',
            max_prompt_tokens: Optional[int] = None,
            embedding=None,
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
        embeddings = np.load(f'data/embeddings/{emb_name}.npy')
        for examples in [self.train, self.dev]:
            for ex in examples:
                ex.embedding = embeddings[ex.datum_id]

    def train_examples(self) -> Iterator[SMCExample]:
        for example in self.train:
            yield example

    def dev_examples(self) -> Iterator[SMCExample]:
        for example in self.dev:
            yield example