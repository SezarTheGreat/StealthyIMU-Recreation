import torch
import torchaudio
import speechbrain
import sentencepiece
import numpy
import scipy
import librosa
import pandas

print("--- Environment Check ---")
print(f"PyTorch version: {torch.__version__}")
print(f"Torchaudio version: {torchaudio.__version__}")
print(f"SpeechBrain version: {speechbrain.__version__}")
print(f"SentencePiece version: {sentencepiece.__version__}")
print(f"NumPy version: {numpy.__version__}")
print(f"SciPy version: {scipy.__version__}")
print(f"Librosa version: {librosa.__version__}")
print(f"Pandas version: {pandas.__version__}")

if torch.cuda.is_available():
    print(f"GPU available: YES")
    print(f"GPU name: {torch.cuda.get_device_name(0)}")
    
    # Calculate VRAM in GB
    total_memory = torch.cuda.get_device_properties(0).total_memory
    print(f"GPU VRAM: {total_memory / (1024**3):.2f} GB")
else:
    print(f"GPU available: NO (Using CPU)")
