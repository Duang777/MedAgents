import json
import os
import time
from typing import Any, Dict, List, Optional


class RuleBank:
    def __init__(self, storage_path: str = './memory/rule_bank.json', max_size: int = 500):
        self.storage_path = storage_path
        self.max_size = max_size
        self.rules: List[Dict[str, Any]] = []
        self.load()

    def load(self) -> None:
        if not os.path.exists(self.storage_path):
            self.rules = []
            return
        with open(self.storage_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        self.rules = data if isinstance(data, list) else []

    def save(self) -> None:
        os.makedirs(os.path.dirname(self.storage_path), exist_ok=True)
        with open(self.storage_path, 'w', encoding='utf-8') as f:
            json.dump(self.rules, f, ensure_ascii=False, indent=2)

    def add_rule(
        self,
        rule_text: str,
        trigger_entities: List[str],
        confidence: float,
        embedding: List[float],
    ) -> None:
        item = {
            'id': f"rule_{int(time.time() * 1000)}_{len(self.rules)}",
            'rule_text': rule_text,
            'trigger_entities': trigger_entities,
            'confidence': float(confidence),
            'timestamp': time.time(),
            'usage_count': 0,
            'embedding': embedding,
        }
        self.rules.append(item)
        self.prune(self.max_size)
        self.save()

    def prune(self, max_size: Optional[int] = None) -> None:
        limit = max_size if max_size is not None else self.max_size
        if len(self.rules) <= limit:
            return

        self.rules.sort(
            key=lambda x: float(x.get('confidence', 0.0)) + 0.01 * float(x.get('usage_count', 0)),
            reverse=True,
        )
        self.rules = self.rules[:limit]
