import sys
import ast

original_eval = ast.literal_eval

def safe_literal_eval(node_or_string):
    try:
        res = original_eval(node_or_string)
        import inspect
        frame = inspect.currentframe().f_back
        if frame and frame.f_code.co_name == "compute_objectives":
            if not isinstance(res, dict):
                raise SyntaxError("Not a dict")
        return res
    except Exception as e:
        import inspect
        frame = inspect.currentframe().f_back
        if frame and frame.f_code.co_name == "compute_objectives":
            raise SyntaxError("Not a dict")
        raise e

ast.literal_eval = safe_literal_eval

# Ensure run_training is called with the right args
# Replace the first arg (run_training.py) with train.py so SpeechBrain argument parser doesn't get confused
sys.argv[0] = "train.py"

with open("train.py", "r", encoding="utf-8") as f:
    code = compile(f.read(), "train.py", "exec")
    exec(code, {"__name__": "__main__"})
