from typing import List
from dataset import SMCExample
from converters.base_converter import BaseConverter
from converters.registry import register_converter


@register_converter('question_only')
class QuestionOnlyConverter(BaseConverter):
    """
    ```
    >>> from converters.registry import get_converter
    >>> converter = get_converter('question_only')
    >>> print(converter.example2code(example))
    Question: what is the brand of this camera?
    Answer: dakota
    >>> print(converter.example2code(example))
    Question: what is the brand of this camera?
    Answer:
    ```
    """

    def example2code(self, demos: List[SMCExample], target: SMCExample) -> str:
        rst = ''
        for example in demos:
            rst += f"source: {example.user_utterance}\n"
            rst += f"target: {example.agent_utterance}\n"
        rst += f"source: {target.user_utterance}\n"
        rst += f"target:"
        return rst

    def code2answer(self, code: str) -> str:
        lines = code.strip().split('\n')
        targets = [line for line in lines if line.startswith('target')]
        return targets[-1].replace('target:', '').strip()
