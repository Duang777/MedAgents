from typing import Optional


class ReflectionAgent:
    def __init__(self, handler):
        self.handler = handler

    def reflect(self, question: str, reasoning_trace: str, outcome: str) -> Optional[str]:
        if outcome != 'success':
            return None

        system_role = (
            'You are a senior clinical reasoning mentor. '
            'Extract one reusable, case-agnostic diagnostic strategy from the successful reasoning trace.'
        )
        prompt = (
            'Please extract a concise and reusable clinical reasoning principle from the following successful case.\n'
            'The principle must be transferable to future cases and should not rely on patient-specific identifiers.\n\n'
            f'Question:\n{question}\n\n'
            f'Reasoning Trace:\n{reasoning_trace}\n\n'
            'Output format:\n'
            'Reflection: [one reusable strategy in 2-4 sentences]'
        )
        reflection = self.handler.get_output_multiagent(
            user_input=prompt,
            temperature=0,
            max_tokens=220,
            system_role=system_role,
        )
        if reflection == 'ERROR.':
            return None
        return reflection
