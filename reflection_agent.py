import json
from typing import Any, Dict, List

from dual_stage_retriever import extract_entities


class EvolutionReflector:
    def __init__(self, handler, rule_bank, confidence_threshold: float = 0.9):
        self.handler = handler
        self.rule_bank = rule_bank
        self.confidence_threshold = confidence_threshold

    def reflect_and_update(
        self,
        question: str,
        reasoning_trace: str,
        gold_answer: str,
        pred_answer: str,
        confidence: float,
    ) -> List[str]:
        if pred_answer != gold_answer:
            return []
        if confidence < self.confidence_threshold:
            return []

        system_role = (
            'You are a senior clinical reasoning mentor. '
            'Extract 1-2 reusable and case-agnostic diagnostic rules from a successful case.'
        )
        prompt = (
            'From the successful diagnostic case below, extract 1-2 reusable diagnostic rules that do not depend on patient-specific details.\n\n'
            f'Question:\n{question}\n\n'
            f'Reasoning Trace:\n{reasoning_trace}\n\n'
            f'Correct Answer:\n{gold_answer}\n\n'
            'Output strict JSON only:\n'
            '{"rules": ["rule 1", "rule 2"]}'
        )
        output = self.handler.get_output_multiagent(
            user_input=prompt,
            temperature=0,
            max_tokens=300,
            system_role=system_role,
        )
        if output == 'ERROR.':
            return []

        rules = self._parse_rules(output)
        for rule in rules:
            emb = self.handler.get_embedding(rule)
            if not emb:
                continue
            self.rule_bank.add_rule(
                rule_text=rule,
                trigger_entities=extract_entities(rule),
                confidence=confidence,
                embedding=emb,
            )
        return rules

    def _parse_rules(self, output: str) -> List[str]:
        text = output.strip()
        try:
            payload = json.loads(text)
            rules = payload.get('rules', [])
            if isinstance(rules, list):
                return [str(r).strip() for r in rules if str(r).strip()]
        except Exception:
            pass

        start = text.find('[')
        end = text.rfind(']')
        if start != -1 and end != -1 and end > start:
            snippet = text[start:end + 1]
            try:
                arr = json.loads(snippet)
                if isinstance(arr, list):
                    return [str(r).strip() for r in arr if str(r).strip()]
            except Exception:
                return []
        return []
