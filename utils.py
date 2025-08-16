import os, sys, re, tempfile, subprocess, random
from typing import Tuple
import torch

# 시드 고정
def set_seed(seed: int):
    random.seed(seed)
    try:
        import numpy as np
        np.random.seed(seed)
    except Exception:
        pass
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

# dtype 선택
def pick_dtype(name: str):
    name = (name or "auto").lower()
    if name == "auto":
        return torch.bfloat16 if torch.cuda.is_available() else torch.float32
    if name == "bfloat16":
        return torch.bfloat16
    if name in ("float16", "fp16"):
        return torch.float16
    return torch.float32

# 디바이스
def to_device():
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 코드펜스 제거/보정
_CODE_FENCE_RE = re.compile(r"```(?:python)?\s*(.*?)\s*```", re.DOTALL)

def strip_code_fences(txt: str) -> str:
    m = _CODE_FENCE_RE.findall(txt)
    return m[0].strip() if m else txt.strip()

def ensure_contains_def(generated: str, fn_name: str) -> str:
    pat = re.compile(rf"def\s+{re.escape(fn_name)}\s*\(")
    m = pat.search(generated)
    if not m:
        return generated
    return generated[m.start():].strip()

# 타임아웃 실행
def run_python_with_timeout(code_str: str, timeout_sec: int) -> Tuple[bool, str]:
    with tempfile.NamedTemporaryFile("w", suffix=".py", delete=False) as f:
        f.write(code_str)
        path = f.name
    try:
        proc = subprocess.run(
            [sys.executable, path],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            timeout=timeout_sec,
            text=True
        )
        ok = (proc.returncode == 0)
        return ok, (proc.stdout or "") + (proc.stderr or "")
    except subprocess.TimeoutExpired:
        return False, f"[TIMEOUT] {timeout_sec}s"
    finally:
        try:
            os.remove(path)
        except Exception:
            pass
