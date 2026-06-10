import subprocess
import sys
import datetime

def run_cmd(cmd):
    print(f"[{datetime.datetime.now()}] Running: {cmd}")
    result = subprocess.run(cmd, shell=True)
    if result.returncode != 0:
        print(f"Command failed with exit code {result.returncode}")
        sys.exit(result.returncode)
    print(f"[{datetime.datetime.now()}] Finished successfully: {cmd}")

if __name__ == "__main__":
    print("=== STARTING PHASE 1: RUN 1 (open_source.yaml) ===")
    run_cmd("venv\\Scripts\\python.exe run_training.py hparams/open_source.yaml --device cpu")
    
    print("=== STARTING PHASE 1: RUN 2 (paper_exact.yaml) ===")
    run_cmd("venv\\Scripts\\python.exe run_training.py hparams/paper_exact.yaml --device cpu")
    
    print("=== PHASE 1 TRAINING COMPLETE ===")
