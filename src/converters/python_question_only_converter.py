from typing import List
from dataset import SMCExample
from converters.base_converter import BaseConverter
from converters.registry import register_converter


@register_converter('python_question_only')
class PythonQuestionOnlyConverter(BaseConverter):
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
        rst = "# Complete the following code, do not start from the beginning of the code.\n"
        rst += "dataset = [\n"
        for example in demos:
            rst+=f"\tExample(question=\'{example.user_utterance}\',answer=\'{example.agent_utterance}\'),\n"
        rst += f"\tExample(question=\'{target.user_utterance}\',answer='"
        return rst

    def code2answer(self, code: str) -> str:
        lines = code.strip().split('\n')
        if 'answer' in lines[-1]:
            line = lines[-1]
        else:
            line = lines[-2]
        answer = line.strip().split('answer=')[-1]
        answer = answer.strip('\'"),')
        return answer