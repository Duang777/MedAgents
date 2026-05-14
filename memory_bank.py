import json
import os
import time
from typing import Any, Dict, List, Optional

from dual_stage_retriever import extract_entities


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
        self.cases = data if isinstance(data, list) else []

    def save(self) -> None:
        os.makedirs(os.path.dirname(self.storage_path), exist_ok=True)
        with open(self.storage_path, 'w', encoding='utf-8') as f:
            json.dump(self.cases, f, ensure_ascii=False, indent=2)

    def add_case(
        self,
        question: str,
        options: Any,
        pred_answer: str,
        gold_answer: str,
        syn_report: str,
        reasoning_trace: str,
        success: bool,
        confidence: float,
        embedding: List[float],
    ) -> None:
        case = {
            'id': f"case_{int(time.time() * 1000)}_{len(self.cases)}",
            'question': question,
            'options': options,
            'pred_answer': pred_answer,
            'gold_answer': gold_answer,
            'syn_report': syn_report,
            'reasoning_trace': reasoning_trace,
            'timestamp': time.time(),
            'success': bool(success),
            'confidence': float(confidence),
            'success_weight': 1.2 if success else 0.7,
            'embedding': embedding,
            'entities': extract_entities(f"{question} {syn_report} {reasoning_trace}"),
        }
        self.cases.append(case)
        self.prune(self.max_size)
        self.save()

    def prune(self, max_size: Optional[int] = None) -> None:
        limit = max_size if max_size is not None else self.max_size
        if len(self.cases) <= limit:
            return

        self.cases.sort(
            key=lambda x: float(x.get('success_weight', 1.0)) * float(x.get('confidence', 0.0)),
            reverse=True,
        )
        self.cases = self.cases[:limit]
