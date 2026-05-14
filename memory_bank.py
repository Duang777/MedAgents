import json
import math
import os
import re
import time
from typing import Any, Dict, List, Optional


class CaseBank:
    def __init__(self, storage_path: str = './memory/case_bank.json', max_size: int = 500):
        self.storage_path = storage_path
        self.max_size = max_size
        self.cases: List[Dict[str, Any]] = []
        self.load()

    def load(self) -> None:
        if not os.path.exists(self.storage_path):
            self.cases = []
            return
        with open(self.storage_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        if isinstance(data, list):
            self.cases = data
        else:
            self.cases = []

    def save(self) -> None:
        os.makedirs(os.path.dirname(self.storage_path), exist_ok=True)
        with open(self.storage_path, 'w', encoding='utf-8') as f:
            json.dump(self.cases, f, ensure_ascii=False, indent=2)

    def _tokenize(self, text: str) -> List[str]:
        text = (text or '').lower()
        text = re.sub(r'[^a-z0-9\s]', ' ', text)
        return [tok for tok in text.split() if tok]

    def _embed(self, text: str) -> Dict[str, float]:
        tokens = self._tokenize(text)
        bag: Dict[str, float] = {}
        for token in tokens:
            bag[token] = bag.get(token, 0.0) + 1.0
        return bag

    def _cosine(self, a: Dict[str, float], b: Dict[str, float]) -> float:
        if not a or not b:
            return 0.0
        common_keys = set(a.keys()) & set(b.keys())
        dot = sum(a[k] * b[k] for k in common_keys)
        na = math.sqrt(sum(v * v for v in a.values()))
        nb = math.sqrt(sum(v * v for v in b.values()))
        if na == 0 or nb == 0:
            return 0.0
        return dot / (na * nb)

    def _time_decay(self, timestamp: float, half_life_days: float = 30.0) -> float:
        age_seconds = max(0.0, time.time() - float(timestamp))
        age_days = age_seconds / 86400.0
        return 0.5 ** (age_days / half_life_days)

    def retrieve(self, question: str, k: int = 3, min_score: float = 0.05) -> List[Dict[str, Any]]:
        query_embedding = self._embed(question)
        scored: List[Dict[str, Any]] = []

        for case in self.cases:
            case_embedding = case.get('embedding') or {}
            sim = self._cosine(query_embedding, case_embedding)
            success_weight = float(case.get('success_weight', 1.0))
            time_weight = self._time_decay(float(case.get('timestamp', time.time())))
            final_score = sim * success_weight * time_weight

            if final_score >= min_score:
                scored.append({'score': final_score, 'case': case})

        scored.sort(key=lambda x: x['score'], reverse=True)
        return [item['case'] for item in scored[:k]]

    def add_case(
        self,
        question: str,
        options: Any,
        pred_answer: str,
        gold_answer: str,
        syn_report: str,
        reasoning_trace: str,
        reflection: Optional[str],
        success: bool,
    ) -> None:
        text_for_embedding = f"{question}\n{reasoning_trace}\n{reflection or ''}"
        case_id = f"case_{int(time.time() * 1000)}_{len(self.cases)}"

        case = {
            'id': case_id,
            'question': question,
            'options': options,
            'pred_answer': pred_answer,
            'gold_answer': gold_answer,
            'syn_report': syn_report,
            'reasoning_trace': reasoning_trace,
            'reflection': reflection or '',
            'timestamp': time.time(),
            'success': bool(success),
            'success_weight': 1.2 if success else 0.7,
            'embedding': self._embed(text_for_embedding),
        }

        self.cases.append(case)
        self.prune(self.max_size)
        self.save()

    def prune(self, max_size: Optional[int] = None) -> None:
        limit = max_size if max_size is not None else self.max_size
        if len(self.cases) <= limit:
            return

        def rank_key(case: Dict[str, Any]) -> float:
            sw = float(case.get('success_weight', 1.0))
            tw = self._time_decay(float(case.get('timestamp', time.time())))
            return sw * tw

        self.cases.sort(key=rank_key, reverse=True)
        self.cases = self.cases[:limit]
