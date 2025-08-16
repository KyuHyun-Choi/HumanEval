from typing import List, Dict

class PromptBuilder:
    @staticmethod
    def build_fewshot_prompt(fewshot_examples: List[Dict], target_prompt: str) -> str:
        examples_text = ""
        for ex in fewshot_examples:
            examples_text += ex["prompt"].strip() + "\n" + ex["canonical_solution"].strip() + "\n\n"
        examples_text += target_prompt.strip() + "\n# Write your solution below."
        return examples_text

    @staticmethod
    def build_chat_prompt_fewshot(fewshot_examples: List[Dict], target_prompt: str) -> List[Dict[str, str]]:
        sys_msg = (
            "You are a highly skilled Python coding assistant. "
            "Write only valid Python code that solves the problem exactly as described."
        )
        user_msg = PromptBuilder.build_fewshot_prompt(fewshot_examples, target_prompt)
        return [
            {"role": "system", "content": sys_msg},
            {"role": "user", "content": user_msg},
        ]

    @staticmethod
    def chat_template(tokenizer, messages: List[Dict[str, str]]) -> str:
        try:
            return tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        except Exception:
            buf = []
            for m in messages:
                buf.append(f"{m['role'].upper()}:\n{m['content']}\n")
            buf.append("ASSISTANT:\n")
            return "\n".join(buf)
