import pandas as pd
import os

base_dir = r"c:\Users\Dilip Kumar Barman\OneDrive\Desktop\StealthyIMU Recreation\results\slu_baseline_repo\1235"
out_dir = r"c:\Users\Dilip Kumar Barman\OneDrive\Desktop\StealthyIMU Recreation\results\sampled_dataset"
os.makedirs(out_dir, exist_ok=True)

# Train
train_df = pd.read_csv(os.path.join(base_dir, "train-type=direct.csv"))
train_df.sample(n=1000, random_state=42).to_csv(os.path.join(out_dir, "train-type=direct.csv"), index=False)

# Valid
valid_df = pd.read_csv(os.path.join(base_dir, "valid-type=direct.csv"))
valid_df.sample(n=200, random_state=42).to_csv(os.path.join(out_dir, "valid-type=direct.csv"), index=False)

# Test
test_df = pd.read_csv(os.path.join(base_dir, "test-type=direct.csv"))
test_df.sample(n=200, random_state=42).to_csv(os.path.join(out_dir, "test-type=direct.csv"), index=False)

print("Dataset sampled successfully into results/sampled_dataset!")
