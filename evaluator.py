import json, textwrap
from typing import Dict, List, Tuple
from datasets import load_dataset
from tqdm.auto import tqdm

from config import Cfg
from utils import run_python_with_timeout
from model import CodeGenerator

class HumanEvalEvaluator:
    def __init__(self, cfg: Cfg, generator: CodeGenerator):
        self.cfg = cfg
        self.generator = generator

    @staticmethod
    def _eval_single_task(solution_code: str, test_code: str, timeout_sec: int) -> Tuple[bool, str]:
        program = solution_code.rstrip() + "\n\n" + test_code.strip() + "\n"
        return run_python_with_timeout(program, timeout_sec)

    def evaluate(self) -> Dict:
        ds = load_dataset("openai_humaneval", split="test")
        fewshot_examples: List[Dict] = [ds[i] for i in range(self.cfg.n_fewshot)]
        eval_tasks: List[Dict] = [ds[i] for i in range(self.cfg.n_fewshot, len(ds))]

        if self.generator.model is None:
            self.generator.load()

        results, n_pass = [], 0
        pbar = tqdm(total=len(eval_tasks), dynamic_ncols=True)

        for i, ex in enumerate(eval_tasks):
            solution = self.generator.generate_solution(fewshot_examples, ex)
            ok, log = self._eval_single_task(solution, ex["test"], self.cfg.timeout_sec)
            if ok:
                n_pass += 1

            results.append({
                "task_id": ex["task_id"],
                "passed": ok,
                "solution": solution,
                "log": (log or "")[:1000]
            })

            pass_rate = n_pass / (i + 1)
            pbar.set_description(f"pass@1: {pass_rate:.4f}")
            pbar.update(1)

            if i < self.cfg.print_examples:
                print("\n=== EXAMPLE ===")
                print("Task:", ex["task_id"])
                print("Prompt:", textwrap.shorten(ex["prompt"], 500))
                print("Solution:", textwrap.shorten(solution, 500))
                print("Result:", "PASS" if ok else "FAIL")
                print("===============")

        pbar.close()
        pass_at_1 = n_pass / len(eval_tasks)

        with open(self.cfg.save_pred_json, "w", encoding="utf-8") as f:
            json.dump(results, f, indent=2, ensure_ascii=False)

        summary = {
            "model_id": self.cfg.model_id,
            "fewshot_examples": self.cfg.n_fewshot,
            "n_eval": len(eval_tasks),
            "pass": n_pass,
            "fail": len(eval_tasks) - n_pass,
            "pass@1": pass_at_1
        }
        with open(self.cfg.save_result_json, "w", encoding="utf-8") as f:
            json.dump(summary, f, indent=2, ensure_ascii=False)

        print("\n=== 결과 ===")
        print(summary)
        return summary
