from dataset import SMCExample
import random
import numpy as np
from typing import List


class BaseDemoSelection:
    def get_demo(self, target: SMCExample) -> List[SMCExample]:
        raise NotImplementedError()


class FixedRandomDemoSelection(BaseDemoSelection):
    # demo_question_ids = [
    #     15283,
    #     25051,
    #     12056,
    #     15883,
    #     33293
    # ]

    def __init__(self, examples: List[SMCExample], n_shot: int = 5):
        assert isinstance(examples, list)
        assert len(examples) >= n_shot

        self.n_shot = n_shot

        qid2example = {ex.datum_id: ex for ex in examples}

        # if n_shot == len(self.demo_question_ids):
        #     self.demonstrations = [qid2example[qid] for qid in self.demo_question_ids]
        # else:
        self.demonstrations = random.sample(examples, n_shot)

    def get_demo(self, target: SMCExample) -> List[SMCExample]:
        return self.demonstrations


class L2TopKDemoSelection(BaseDemoSelection):
    def __init__(self, examples: List[SMCExample], n_shot: int = 5):
        assert isinstance(examples, list)
        assert len(examples) >= n_shot

        self.examples = examples
        self.n_shot = n_shot
        self.X_emb = np.array([ex.utterance_emb for ex in examples], dtype=np.float32)

    def get_demo(self, target: SMCExample) -> List[SMCExample]:
        assert isinstance(target.utterance_emb, np.ndarray)
        X = self.X_emb
        y = target.utterance_emb
        # dist[i] = |X[i] - y|^2
        #         = |X[i]|^2 - 2 * X[i]^T y + |y|^2
        #         ~ |X[i]|^2 - 2 * X[i]^T y
        dist = (X ** 2).sum(1) - 2 * X.dot(y)
        demo_ids = np.argsort(dist)[:self.n_shot]
        return [self.examples[i] for i in demo_ids[::-1]]


class CosineTopKDemoSelection(BaseDemoSelection):
    def __init__(self, examples: List[SMCExample], n_shot: int = 5):
        assert isinstance(examples, list)
        assert len(examples) >= n_shot

        self.examples = examples
        self.n_shot = n_shot
        self.X_emb = np.array([ex.utterance_emb for ex in examples], dtype=np.float32)

    def get_demo(self, target: SMCExample) -> List[SMCExample]:
        assert isinstance(target.utterance_emb, np.ndarray)
        X = self.X_emb
        y = target.utterance_emb
        # dist[i] = -cosine(X[i], y)
        #         = - X[i]^T y / |X[i]| |y|
        #         ~ - X[i]^T y / |X[i]|
        dist = -X.dot(y) / np.sqrt((X ** 2).sum(1))
        demo_ids = np.argsort(dist)[:self.n_shot]
        return [self.examples[i] for i in demo_ids[::-1]]