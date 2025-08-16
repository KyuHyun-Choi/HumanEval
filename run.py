import os, sys, warnings
from config import Cfg
from utils import set_seed
from model import CodeGenerator
from evaluator import HumanEvalEvaluator

os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
os.environ.setdefault("TRANSFORMERS_NO_TF", "1")
os.environ.setdefault("TRANSFORMERS_NO_FLAX", "1")
warnings.filterwarnings("ignore")

class Runner:
    def __init__(self, cfg: Cfg | None = None):
        self.cfg = cfg or Cfg()

    def run(self):
        set_seed(self.cfg.seed)
        generator = CodeGenerator(self.cfg)
        evaluator = HumanEvalEvaluator(self.cfg, generator)
        evaluator.evaluate()

if __name__ == "__main__":
    Runner().run()
    
