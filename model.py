from typing import Dict, List, Tuple
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from config import Cfg
from prompt import PromptBuilder
from utils import pick_dtype, to_device, strip_code_fences, ensure_contains_def

class CodeGenerator:
    def __init__(self, cfg: Cfg):
        self.cfg = cfg
        self.device = to_device()
        self.model = None
        self.tok = None

    def load(self):
        torch_dtype = pick_dtype(self.cfg.dtype)
        tok = AutoTokenizer.from_pretrained(self.cfg.model_id, trust_remote_code=True)
        if tok.pad_token is None:
            tok.pad_token = tok.eos_token or tok.unk_token
        model = AutoModelForCausalLM.from_pretrained(
            self.cfg.model_id,
            torch_dtype=torch_dtype,
            device_map="auto" if torch.cuda.is_available() else None,
            trust_remote_code=True
        )
        model.eval()
        self.model, self.tok = model, tok

    def generate_solution(self, fewshot_examples: List[Dict], task: Dict) -> str:
        msgs = PromptBuilder.build_chat_prompt_fewshot(fewshot_examples, task["prompt"])
        prompt_text = PromptBuilder.chat_template(self.tok, msgs)
        inputs = self.tok(prompt_text, return_tensors="pt").to(self.device)
        with torch.no_grad():
            gen = self.model.generate(
                **inputs,
                max_new_tokens=self.cfg.max_new_tokens,
                do_sample=self.cfg.do_sample,
                temperature=self.cfg.temperature,
                top_p=self.cfg.top_p,
                eos_token_id=self.tok.eos_token_id,
                pad_token_id=self.tok.pad_token_id
            )
        out = self.tok.decode(gen[0], skip_special_tokens=True)
        gen_only = out.split(prompt_text)[-1].strip()
        code = strip_code_fences(gen_only)
        code = ensure_contains_def(code, task["entry_point"])
        return code
