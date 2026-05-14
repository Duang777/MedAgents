import math
import re
from typing import Any, Dict, List, Tuple


def cosine(a: List[float], b: List[float]) -> float:
    if not a or not b or len(a) != len(b):
        return 0.0
    dot = sum(x * y for x, y in zip(a, b))
    na = math.sqrt(sum(x * x for x in a))
    nb = math.sqrt(sum(y * y for y in b))
    if na == 0.0 or nb == 0.0:
        return 0.0
    return dot / (na * nb)


def extract_entities(text: str) -> List[str]:
    cleaned = re.sub(r'[^a-zA-Z0-9\s]', ' ', (text or '').lower())
    tokens = [t for t in cleaned.split() if len(t) > 3]
    stop = {
        'that', 'with', 'from', 'this', 'which', 'what', 'when', 'where', 'have', 'will', 'would',
        'there', 'their', 'about', 'into', 'after', 'before', 'under', 'between', 'because',
        'patient', 'patients', 'medical', 'option', 'question', 'report', 'analysis'
    }
    return [t for t in tokens if t not in stop][:30]


def jaccard(a: List[str], b: List[str]) -> float:
    sa, sb = set(a), set(b)
    if not sa or not sb:
        return 0.0
    return len(sa & sb) / len(sa | sb)


class DualStageRetriever:
    def __init__(
        self,
        case_bank,
        rule_bank,
        api_handler,
        familiarity_top_m: int = 15,
        familiarity_threshold: float = 0.6,
        w1: float = 0.5,
        w2: float = 0.3,
        w3: float = 0.2,
    ):
        self.case_bank = case_bank
        self.rule_bank = rule_bank
        self.api_handler = api_handler
        self.familiarity_top_m = familiarity_top_m
        self.familiarity_threshold = familiarity_threshold
        self.w1 = w1
        self.w2 = w2
        self.w3 = w3

    def _path_proxy(self, question: str, case: Dict[str, Any]) -> float:
        q_tokens = extract_entities(question)
        trace_tokens = extract_entities(case.get('reasoning_trace', ''))
        return jaccard(q_tokens, trace_tokens)

    def retrieve_cases(self, question: str, top_k: int = 3) -> List[Dict[str, Any]]:
        query_emb = self.api_handler.get_embedding(question)
        if not query_emb:
            return []

        stage1: List[Tuple[float, Dict[str, Any]]] = []
        for case in self.case_bank.cases:
            emb = case.get('embedding') or []
            sem = cosine(query_emb, emb)
            if sem >= self.familiarity_threshold:
                stage1.append((sem, case))

        stage1.sort(key=lambda x: x[0], reverse=True)
        candidates = stage1[: self.familiarity_top_m]

        q_entities = extract_entities(question)
        reranked: List[Tuple[float, Dict[str, Any]]] = []
        for sem, case in candidates:
            path_sim = self._path_proxy(question, case)
            ent_sim = jaccard(q_entities, case.get('entities', []))
            score = self.w1 * sem + self.w2 * path_sim + self.w3 * ent_sim
            reranked.append((score, case))

        reranked.sort(key=lambda x: x[0], reverse=True)
        return [x[1] for x in reranked[:top_k]]

    def retrieve_rules(self, question: str, top_kr: int = 2) -> List[Dict[str, Any]]:
        query_emb = self.api_handler.get_embedding(question)
        if not query_emb:
            return []

        q_entities = extract_entities(question)
        scored: List[Tuple[float, Dict[str, Any]]] = []
        for rule in self.rule_bank.rules:
            sem = cosine(query_emb, rule.get('embedding', []))
            ent = jaccard(q_entities, rule.get('trigger_entities', []))
            score = 0.7 * sem + 0.3 * ent
            scored.append((score, rule))

        scored.sort(key=lambda x: x[0], reverse=True)
        return [x[1] for x in scored[:top_kr]]
