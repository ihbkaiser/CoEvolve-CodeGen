from .Dataset import Dataset
from constants.paths import *
from utils.evaluate_lcb import evaluate
import json
class LCBDataset(Dataset):
    def __init__(
        self,
        path: str = LCB_DATA_PATH,
    ):
        super().__init__(path)
        self.id_key = "problem_id"
    
    def evaluate(
        self,
        item: dict,
        cur_imp: str,
        language: str,
    ):
        logs, passed = evaluate(
            code=cur_imp,
            test_cases=json.loads(item["private_test_cases"])
        )
        return passed

    def evaluate_sample_io(
        self,
        item: dict,
        cur_imp: str,
        language: str,
    ):
        
        logs, passed = evaluate(
            code=cur_imp,
            test_cases=json.loads(item["public_test_cases"])
        )
        return passed, logs
    def get_sample_io(
        self,
        item: dict,
    ):
        sample_io_str = ""
        for io in json.loads(item["public_test_cases"]):
            sample_io_str += f"Input:\n{io['input']}\nExpected output:\n{io['output']}\n"
        return sample_io_str

    @staticmethod
    def get_prompt(item):
        return item["question_content"]