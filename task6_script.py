import sentencepiece as spm
import numpy as np
import torch
import sys
from hyperpyyaml import load_hyperpyyaml

def task6a():
    print("--- Step 6a: Tokenizer ---")
    sp = spm.SentencePieceProcessor(model_file='pretrain/51_unigram.model')
    sample_semantics = "{'action': 'air'| 'entities': [{'type': 'city'| 'filler': 'los angeles'}]}"
    encoded = sp.encode_as_ids(sample_semantics)
    decoded = sp.decode_ids(encoded)
    print(f"Original: {sample_semantics}")
    print(f"Encoded IDs: {encoded}")
    print(f"Decoded: {decoded}")
    assert decoded.replace(" ", "") == sample_semantics.replace(" ", ""), "Mismatch in decoded string"
    print("Round trip works perfectly.")

def task6b():
    print("\n--- Step 6b: Data Loader Logic ---")
    # Read first 5 lines of the CSV
    with open('StealthyIMU_dataset/metadata/stealthyIMU_all_relative.csv', 'r', encoding='utf-8') as f:
        lines = [next(f) for x in range(5)]
    
    shapes = []
    for line in lines:
        parts = line.strip().split(',')
        if len(parts) < 3: continue
        wav_path = parts[2] # e.g. ./data/cleanair/.../uuid.wav
        # train.py logic:
        # filename = os.path.join(hparams["data_folder"], wav) -> wav path already contains ./data/
        # Wait, if wav path is ./data/..., and data_folder is StealthyIMU_dataset/,
        # os.path.join("StealthyIMU_dataset/", "./data/...") = StealthyIMU_dataset/data/...
        import os
        filename = os.path.join("StealthyIMU_dataset", wav_path)
        accnpy_path = filename[:-4] + '.accnpy'
        if not os.path.exists(accnpy_path):
            print(f"File not found: {accnpy_path}")
            continue
        signal = np.load(accnpy_path)
        signal = signal[3,:]
        signal = torch.from_numpy(signal).float().to('cpu')
        shapes.append(signal.shape)
        print(f"Loaded {accnpy_path}, shape: {signal.shape}")
    
    # check consistency
    print(f"Shapes match: {len(set(shapes)) == 1}")

def task6c():
    print("\n--- Step 6c: Instantiate Model ---")
    with open('hparams/open_source.yaml') as fin:
        hparams = load_hyperpyyaml(fin)
    
    # Need to setup model logic
    # The dummy batch should just be features
    print("Model loaded successfully")
    enc = hparams["enc"]
    compute_features = hparams["compute_features"]
    
    # Dummy waveform (B=1, T=3471)
    wav = torch.randn(1, 3471)
    wav_lens = torch.tensor([1.0])
    
    feats = compute_features(wav)
    print(f"Features shape: {feats.shape}")
    
    encoder_out = enc(feats)
    print(f"Encoder out shape: {encoder_out.shape}")

if __name__ == "__main__":
    task6a()
    task6b()
    task6c()
