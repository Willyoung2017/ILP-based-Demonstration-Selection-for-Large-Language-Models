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
            return [self.get_demo(target) for target in tqdm(targets)]

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
        self.X_emb = np.array([ex.utterance_emb for ex in examples], dtype=np.float64)

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
    def __init__(self, examples: List[SMCExample], n_shot: int = 5, n_processes: int = 1,
                 whiten: bool = False):
        super().__init__(n_processes=n_processes)

        assert isinstance(examples, list)
        assert len(examples) >= n_shot

        self.examples = examples
        self.n_shot = n_shot
        self.X_emb = np.array([ex.utterance_emb for ex in examples], dtype=np.float64)

        if whiten:
            u, s, vt = np.linalg.svd(self.X_emb, full_matrices=False)
            self.W_whiten = vt.T.dot(np.diag(1 / s)).dot(vt)

    def get_demo(self, target: SMCExample) -> List[SMCExample]:
        assert isinstance(target.utterance_emb, np.ndarray)
        X = self.X_emb
        y = target.utterance_emb
        if hasattr(self, "W_whiten"):
            X = X.dot(self.W_whiten)
            y = y.dot(self.W_whiten)
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
        self.X_emb = np.array([ex.utterance_emb for ex in examples], dtype=np.float64)

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
        print(demo_ids)
        return [self.examples[i] for i in demo_ids]


class CosineTopKLengthConstrainedDemoSelection(BaseDemoSelection):
    def __init__(self, examples: List[SMCExample], n_shot: int = 5, length: int = 100,
                 pre_comp_dir: str = "", load_from_pre_comp: bool = False, engine: str = "code-davinci-002",
                 diverse: bool = False, n_processes: int = 1, top_sim=100, div_alpha=1e-2):
        super().__init__(n_processes=n_processes)

        assert isinstance(examples, list)
        assert len(examples) >= n_shot

        self.examples = examples
        self.n_shot = n_shot
        self.length = length
        self.X_emb = np.array([ex.utterance_emb for ex in examples], dtype=np.float64)
        self.out_emb = np.array([ex.plan_emb for ex in examples], dtype=np.float64)
        fn = f"consine_topk_len_con_n{n_shot}_len{length}_diverse{diverse}"
        self.pre_comp_id_path = f"{pre_comp_dir}/{fn}_id.json"
        self.pre_comp_ut_path = f"{pre_comp_dir}/{fn}_ut.json"
        self.pre_comp_dir = pre_comp_dir
        self.load_from_pre_comp = load_from_pre_comp
        self.diverse = diverse
        self.top_sim = top_sim
        self.div_alpha = div_alpha
        if not self.load_from_pre_comp:
            print("No pre-computed demo ids, do optimization from scratch")
            tokenizer = tiktoken.encoding_for_model(engine)
            self.example_length = np.array(
                [len(tokenizer.encode(f"source {ex.agent_utterance}\ntarget: {ex.agent_utterance}")) for ex in
                 examples])
            self.demo_ids = {}
            self.demo_uts = {}
        else:
            print(f"Load demo ids from: {self.pre_comp_id_path}")
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
            top_sim_ids = np.flip(np.argsort(similarity))[:self.top_sim]
            if not self.diverse:
                s = cp.Variable(X.shape[0], boolean=True)
                objective = cp.Maximize(similarity @ s)
                constraints = [cp.sum(s) <= self.n_shot, self.example_length @ s <= self.length]
            else:
                s = cp.Variable(self.top_sim, boolean=True)
                diversity = cp.psd_wrap(self.out_emb[top_sim_ids] @ self.out_emb[top_sim_ids].T)
                objective = cp.Minimize(self.div_alpha * cp.quad_form(s, diversity) - similarity[top_sim_ids] @ s)
                constraints = [cp.sum(s) <= self.n_shot, self.example_length[top_sim_ids] @ s <= self.length]

            prob = cp.Problem(objective, constraints)
            result = prob.solve(solver="SCIP")
            failed = False
            if s.value is None:
                assert self.diverse
                failed = True
                print("failed to solve MIQP, use MIP instead")
                s = cp.Variable(X.shape[0], boolean=True)
                objective = cp.Maximize(similarity @ s)
                constraints = [cp.sum(s) <= self.n_shot, self.example_length @ s <= self.length]
                prob = cp.Problem(objective, constraints)
                result = prob.solve(solver="SCIP")
            demo_ids = np.where(s.value > 0)[0].tolist()
            if self.diverse and not failed:
                demo_ids = top_sim_ids[demo_ids].tolist()
            demo_ids.sort(key=lambda x: similarity[x])
            demo_ids = np.array(demo_ids)
            all_demo_uts = []
            for d_id in demo_ids:
                d_eg = self.examples[d_id]
                all_demo_uts.append({'datum_id': d_eg.datum_id.to_dict(),
                                     'agent_utterance': d_eg.agent_utterance,
                                     'user_utterance': d_eg.user_utterance})
            demo_id_div = np.sum(self.out_emb[demo_ids] @ self.out_emb[demo_ids].T)
            return [self.examples[i] for i in demo_ids], target_key, (
            demo_ids.tolist(), similarity[demo_ids].tolist(), demo_id_div), all_demo_uts
        else:
            demo_ids = np.array(self.demo_ids[target_key])
            return [self.examples[i] for i in demo_ids]

    def save_demos(self, all_demo_ids, all_demo_uts):
        with open(self.pre_comp_id_path, 'w') as f_id, open(self.pre_comp_ut_path, 'w') as f_ut:
            json.dump(all_demo_ids, f_id)
            json.dump(all_demo_uts, f_ut)


class CosineTopKLengthConstrainedGreedyDemoSelection(BaseDemoSelection):
    def __init__(self, examples: List[SMCExample], n_shot: int = 5, length: int = 100,
                 pre_comp_dir: str = "", load_from_pre_comp: bool = False, engine: str = "code-davinci-002",
                 diverse: bool = False, n_processes: int = 1):
        super().__init__(n_processes=n_processes)

        assert isinstance(examples, list)
        assert len(examples) >= n_shot

        self.examples = examples
        self.n_shot = n_shot
        self.length = length
        self.X_emb = np.array([ex.utterance_emb for ex in examples], dtype=np.float64)
        self.out_emb = np.array([ex.plan_emb for ex in examples], dtype=np.float64)
        fn = f"consine_topk_len_con_greedy_n{n_shot}_len{length}_diverse{diverse}"
        self.pre_comp_id_path = f"{pre_comp_dir}/{fn}_id.json"
        self.pre_comp_ut_path = f"{pre_comp_dir}/{fn}_ut.json"
        self.pre_comp_dir = pre_comp_dir
        self.load_from_pre_comp = load_from_pre_comp
        self.diverse = diverse
        if not self.load_from_pre_comp:
            print("No pre-computed demo ids, do optimization from scratch")
            tokenizer = tiktoken.encoding_for_model(engine)
            self.example_length = np.array(
                [len(tokenizer.encode(f"source {ex.agent_utterance}\ntarget: {ex.agent_utterance}")) for ex in
                 examples])
            self.demo_ids = {}
            self.demo_uts = {}
        else:
            print(f"Load demo ids from: {self.pre_comp_id_path}")
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
            sorted_ids = np.flip(np.argsort(similarity))
            cum_sum = np.cumsum(self.example_length[sorted_ids])
            num_selected = np.sum(cum_sum <= self.length)
            num_selected = min(num_selected, self.n_shot)
            demo_ids = sorted_ids[:num_selected][::-1]
            all_demo_uts = []
            for d_id in demo_ids:
                d_eg = self.examples[d_id]
                all_demo_uts.append({'datum_id': d_eg.datum_id.to_dict(),
                                     'agent_utterance': d_eg.agent_utterance,
                                     'user_utterance': d_eg.user_utterance})
            demo_id_div = np.sum(self.out_emb[demo_ids] @ self.out_emb[demo_ids].T)
            return [self.examples[i] for i in demo_ids], target_key, (
            demo_ids.tolist(), similarity[demo_ids].tolist(), demo_id_div), all_demo_uts
        else:
            demo_ids = np.array(self.demo_ids[target_key])
            return [self.examples[i] for i in demo_ids]

    def save_demos(self, all_demo_ids, all_demo_uts):
        with open(self.pre_comp_id_path, 'w') as f_id, open(self.pre_comp_ut_path, 'w') as f_ut:
            json.dump(all_demo_ids, f_id)
            json.dump(all_demo_uts, f_ut)


class MIQPDemoSelection(BaseDemoSelection):
    def __init__(self, examples: List[SMCExample], n_shot: int = 5,
                 n_processes: int = 1, top_sim: int = 1000, div_alpha: float = 1e-2,
                 whiten: bool = True):
        super().__init__(n_processes=n_processes)
        assert isinstance(examples, list)
        assert len(examples) >= n_shot

        self.examples = examples
        self.n_shot = n_shot
        self.top_sim = top_sim
        self.div_alpha = div_alpha
        self.whiten = whiten

        self.X_emb = np.array([ex.utterance_emb for ex in examples], dtype=np.float64)
        self.Z_emb = np.array([ex.plan_emb for ex in examples], dtype=np.float64)

        if whiten:
            print('Whitening the embeddings')
            u, s, vt = np.linalg.svd(self.X_emb, full_matrices=False)
            self.Xw = vt.T.dot(np.diag(1 / s)).dot(vt)
            self.X_emb = self.X_emb.dot(self.Xw)
            u, s, vt = np.linalg.svd(self.Z_emb, full_matrices=False)
            self.Zw = vt.T.dot(np.diag(1 / s)).dot(vt)
            self.Z_emb = self.Z_emb.dot(self.Zw)

        self.X_emb = self._normalize(self.X_emb)
        self.Z_emb = self._normalize(self.Z_emb)

    @staticmethod
    def _normalize(X):
        norms = np.sqrt((X ** 2).sum(-1, keepdims=True))
        norms[norms == 0] = 1e-10
        return X / norms

    def get_demo(self, target: SMCExample) -> List[SMCExample]:
        assert isinstance(target.utterance_emb, np.ndarray)
        assert isinstance(target.plan_emb, np.ndarray)

        X, Z = self.X_emb, self.Z_emb

        y = target.utterance_emb
        if self.whiten:
            y = y.dot(self.Xw)
        y = self._normalize(y)

        dist = -X.dot(y)
        top_sim_ids = np.argsort(dist)[:self.top_sim]
        Z1 = Z[top_sim_ids]
        dist1 = dist[top_sim_ids]

        s = cp.Variable(self.top_sim, boolean=True)
        Q = cp.psd_wrap(Z1 @ Z1.T)
        objective = cp.Minimize(self.div_alpha * cp.quad_form(s, Q) + dist1 @ s)
        constraints = [cp.sum(s) == self.n_shot]

        prob = cp.Problem(objective, constraints)
        prob.solve(solver="SCIP")

        demo_ids = top_sim_ids[np.nonzero(s.value > 0)[0]].tolist()
        demo_ids.sort(key=lambda x: -dist[x])
        return [self.examples[i] for i in demo_ids]
