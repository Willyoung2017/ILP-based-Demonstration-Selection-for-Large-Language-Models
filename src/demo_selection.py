from dataset import SMCExample
import random
import numpy as np
from typing import List
import cvxpy as cp
import tiktoken
import multiprocessing as mp
from tqdm import tqdm
import json


class BaseDemoSelection:
    def __init__(self, n_processes: int = 1):
        self.n_processes = n_processes

    def get_demo(self, target: SMCExample) -> List[SMCExample]:
        raise NotImplementedError()

    def batch_get_demo(self, targets: List[SMCExample]) -> List[List[SMCExample]]:
        if self.n_processes <= 1:
            return [self.get_demo(target) for target in targets]

        with mp.Pool(self.n_processes) as pool:
            return list(tqdm(
                pool.imap(self.get_demo, targets),
                disable=False, total=len(targets)
            ))


class FixedRandomDemoSelection(BaseDemoSelection):
    # demo_question_ids = [
    #     15283,
    #     25051,
    #     12056,
    #     15883,
    #     33293
    # ]

    def __init__(self, examples: List[SMCExample], n_shot: int = 5, n_processes: int = 1):
        super().__init__(n_processes=n_processes)

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
    def __init__(self, examples: List[SMCExample], n_shot: int = 5, n_processes: int = 1):
        super().__init__(n_processes=n_processes)

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
    def __init__(self, examples: List[SMCExample], n_shot: int = 5, n_processes: int = 1):
        super().__init__(n_processes=n_processes)

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


class IPCosineTopKDemoSelection(BaseDemoSelection):
    def __init__(self, examples: List[SMCExample], n_shot: int = 5, n_processes: int = 1):
        super().__init__(n_processes=n_processes)

        assert isinstance(examples, list)
        assert len(examples) >= n_shot

        self.examples = examples
        self.n_shot = n_shot
        self.X_emb = np.array([ex.utterance_emb for ex in examples], dtype=np.float32)

    def get_demo(self, target: SMCExample) -> List[SMCExample]:
        assert isinstance(target.utterance_emb, np.ndarray)
        X = self.X_emb
        y = target.utterance_emb
        similarity = X.dot(y) / np.sqrt((X ** 2).sum(1))
        s = cp.Variable(X.shape[0], boolean=True)
        constraints = [cp.sum(s) == self.n_shot]
        objective = cp.Maximize(similarity @ s)
        prob = cp.Problem(objective, constraints)
        prob.solve(solver="SCIP")
        demo_ids = np.nonzero(s.value > 0)[0]
        assert demo_ids.shape[0] == self.n_shot
        demo_ids = sorted(demo_ids, key=lambda i: similarity[i])
        return [self.examples[i] for i in demo_ids]


class CosineTopKLengthConstrainedDemoSelection(BaseDemoSelection):
    def __init__(self, examples: List[SMCExample], n_shot: int = 5, length: int = 100,
                 pre_comp_dir: str = "", load_from_pre_comp: bool = False, engine: str = "code-davinci-002",
                 diverse: bool = False, n_processes: int = 1):
        super().__init__(n_processes=n_processes)

        assert isinstance(examples, list)
        assert len(examples) >= n_shot

        self.examples = examples
        self.n_shot = n_shot
        self.length = length
        self.X_emb = np.array([ex.utterance_emb for ex in examples], dtype=np.float32)
        fn = f"consine_topk_len_con_n{n_shot}_len{length}_diverse{diverse}"
        self.pre_comp_id_path = f"{pre_comp_dir}/{fn}_id.json"
        self.pre_comp_ut_path = f"{pre_comp_dir}/{fn}_ut.json"
        self.pre_comp_dir = pre_comp_dir
        self.load_from_pre_comp = load_from_pre_comp
        self.diverse = diverse
        if not self.load_from_pre_comp:
            print("No pre-computed demo ids, do optimization from scratch")
            self.tokenizer = tiktoken.encoding_for_model(engine)
            self.example_length = np.array(
                [len(self.tokenizer.encode(f"source {ex.agent_utterance}\ntarget: {ex.agent_utterance}")) for ex in
                 examples])
            self.demo_ids = {}
            self.demo_uts = {}
        else:
            print(f"Load demo ids from: {self.pre_comp_id_path}")
            self.tokenizer = None
            self.example_length = None
            self.demo_ids = json.load(open(self.pre_comp_id_path, 'r'))
            self.demo_uts = json.load(open(self.pre_comp_ut_path, 'r'))

    def get_demo(self, target: SMCExample) -> List[SMCExample]:
        assert isinstance(target.utterance_emb, np.ndarray)
        target_key = f"{target.datum_id.dialogue_id}-{target.datum_id.turn_index}"
        if not self.load_from_pre_comp:
            X = self.X_emb
            y = target.utterance_emb
            similarity = X.dot(y) / np.sqrt((X ** 2).sum(1))
            s = cp.Variable(X.shape[0], boolean=True)
            constraints = [cp.sum(s) <= self.n_shot, self.example_length @ s <= self.length]
            if not self.diverse:
                objective = cp.Maximize(similarity @ s)
            else:
                diversity = X @ X.T
                objective = cp.Minimize(s @ diversity @ s.T - similarity @ s)
            prob = cp.Problem(objective, constraints)
            result = prob.solve(solver="SCIP")
            demo_ids = np.where(s.value > 0)[0].tolist()
            demo_ids.sort(key=lambda x: similarity[x])
            demo_ids = np.array(demo_ids)
            self.demo_ids[target_key] = demo_ids.tolist()
            self.demo_uts[target_key] = []
            for d_id in demo_ids:
                d_eg = self.examples[d_id]
                self.demo_uts[target_key].append({'datum_id': d_eg.datum_id.to_dict(),
                                                  'agent_utterance': d_eg.agent_utterance,
                                                  'user_utterance': d_eg.user_utterance})
            return [self.examples[i] for i in demo_ids]
        else:
            demo_ids = np.array(self.demo_ids[target_key])
            return [self.examples[i] for i in demo_ids]

    def save_demos(self):
        with open(self.pre_comp_id_path, 'w') as f_id, open(self.pre_comp_ut_path, 'w') as f_ut:
            json.dump(self.demo_ids, f_id)
            json.dump(self.demo_uts, f_ut)
