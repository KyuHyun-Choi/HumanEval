from dataclasses import dataclass

@dataclass
class Cfg:
    # 모델 및 디코딩
    model_id: str = "deepseek-ai/deepseek-coder-6.7b-instruct"
    dtype: str = "auto"
    max_new_tokens: int = 384
    temperature: float = 0.0
    top_p: float = 1.0
    do_sample: bool = False

    # 평가 세팅
    n_fewshot: int = 0
    timeout_sec: int = 8
    seed: int = 42
    print_examples: int = 2

    # 저장 경로
    save_pred_json: str = "./humaneval_deepseek6.7b_fewshot_predictions.json"
    save_result_json: str = "./humaneval_deepseek6.7b_fewshot_results.json"
