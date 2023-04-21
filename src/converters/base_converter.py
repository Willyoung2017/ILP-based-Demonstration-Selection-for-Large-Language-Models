from typing import List
from dataset import SMCExample

class BaseConverter:
    def example2code(self, demos: List[SMCExample], target: SMCExample) -> str:
        raise NotImplementedError()

    def code2answer(self, code: str) -> str:
        raise NotImplementedError()