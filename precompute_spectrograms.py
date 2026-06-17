import os
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset

class AccSpec(nn.Module):
    """
    Replicated feature extractor matching feature.py/SpeechBrain's STFT configuration.
    """
    def __init__(
        self,
        sample_rate=500,
        n_fft=80,
        win_length=80,
        hop_length=20,
        left_frames=5,
        right_frames=5,
        context=False,
    ):
        super().__init__()
        self.sample_rate = sample_rate
        self.n_fft = n_fft
        self.win_length = win_length
        self.hop_length = hop_length
        self.context = context
        
        # Match SpeechBrain's windowing function (standard Hamming window)
        self.register_buffer("window", torch.hamming_window(self.win_length))

    def forward(self, wav):
        """
        Computes spectrogram matching SpeechBrain AccSpec feature extractor.
        
        Arguments
        ---------
        wav : torch.Tensor
            Input signals of shape (B, Time_Steps)
        
        Returns
        -------
        spectrogram : torch.Tensor
            Spectrogram of shape (B, Time_Frames, Freq_Bins_Truncated)
            Truncation keeps frequencies starting at index 10 (freq bins: 10 to 40, total 31 bins)
        """
        # torch.stft returns complex tensor of shape (B, Freq_Bins, Time_Frames, 2) or complex tensor depending on version
        stft_res = torch.stft(
            wav,
            n_fft=self.n_fft,
            hop_length=self.hop_length,
            win_length=self.win_length,
            window=self.window,
            center=True,
            pad_mode="reflect",
            normalized=False,
            onesided=True,
            return_complex=True,
        )
        
        # Spectral magnitude: sqrt(real^2 + imag^2)
        mag = torch.abs(stft_res)
        
        # Transpose to shape (B, Time_Frames, Freq_Bins) to match SpeechBrain output format
        mag = mag.transpose(1, 2)
        
        # Truncate to index 10: (matches mag[:,:,10:] yielding 31 frequency bins)
        return mag[:, :, 10:]


def precompute_features(
    raw_data,
    sample_rate=500,
    n_fft=80,
    win_length=80,
    hop_length=20,
    channel_axis=-1,  # -1 if (N, Time, Channels), 1 if (N, Channels, Time)
    batch_size=128,
):
    """
    Computes spectrogram features for a multi-channel raw motion signal dataset.
    
    Arguments
    ---------
    raw_data : np.ndarray or torch.Tensor
        Raw motion sensor dataset of shape (N, Time_Steps, Channels) or (N, Channels, Time_Steps).
    sample_rate : int
        Sampling rate in Hz (500 Hz).
    n_fft : int
        FFT size (80).
    win_length : int
        Window length (80).
    hop_length : int
        Hop length (20).
    channel_axis : int
        The dimension index corresponding to the Channels.
    batch_size : int
        Batch size to process features efficiently on CPU/GPU.
        
    Returns
    -------
    features : np.ndarray
        Consolidated spectrogram features of shape (N, Time_Frames, Freq_Bins_31, Channels).
    """
    if isinstance(raw_data, np.ndarray):
        raw_data = torch.from_numpy(raw_data).float()
    else:
        raw_data = raw_data.float()
        
    # Convert data shape to standard (N, Channels, Time_Steps)
    if channel_axis == -1 or channel_axis == 2:
        raw_data = raw_data.permute(0, 2, 1)  # (N, Channels, Time_Steps)
        
    N, C, T = raw_data.shape
    spec_extractor = AccSpec(
        sample_rate=sample_rate,
        n_fft=n_fft,
        win_length=win_length,
        hop_length=hop_length,
    )
    
    # Process batch-wise to manage memory overhead
    all_spec_features = []
    
    for i in range(0, N, batch_size):
        batch = raw_data[i : i + batch_size]  # (B, Channels, Time_Steps)
        B = batch.shape[0]
        
        # Flatten (B, Channels, Time_Steps) to (B * Channels, Time_Steps) to compute STFT in parallel
        batch_flattened = batch.reshape(B * C, T)
        
        with torch.no_grad():
            # (B * Channels, Time_Frames, Freq_Bins_31)
            feats = spec_extractor(batch_flattened)
            
        _, Time_Frames, Freq_Bins = feats.shape
        # Reshape back to (B, Channels, Time_Frames, Freq_Bins_31)
        feats = feats.reshape(B, C, Time_Frames, Freq_Bins)
        # Permute to (B, Time_Frames, Freq_Bins_31, Channels)
        feats = feats.permute(0, 2, 3, 1)
        
        all_spec_features.append(feats.numpy())
        
    # Consolidate all batches
    consolidated_features = np.concatenate(all_spec_features, axis=0)
    return consolidated_features


class PrecomputedDataset(Dataset):
    """
    A lightweight, high-performance PyTorch Dataset that loads precomputed spectrogram features 
    directly from disk using memory-mapping (mmap) or direct RAM loading.
    """
    def __init__(self, features_path, labels_path=None, use_mmap=True):
        """
        Arguments
        ---------
        features_path : str
            Path to the precomputed .npy file (e.g., 'x_train_spec.npy').
        labels_path : str, optional
            Path to target labels/transcripts .npy file or similar.
        use_mmap : bool
            If True, uses numpy memory-mapping (mmap_mode='r') to keep RAM utilization near 0 
            and load slices dynamically on-the-fly. If False, loads the entire array into RAM.
        """
        if not os.path.exists(features_path):
            raise FileNotFoundError(f"Features file not found at: {features_path}")
            
        if use_mmap:
            self.features = np.load(features_path, mmap_mode="r")
        else:
            self.features = np.load(features_path)
            
        self.labels = None
        if labels_path is not None:
            self.labels = np.load(labels_path)

    def __len__(self):
        return self.features.shape[0]

    def __getitem__(self, idx):
        # Fetch feature slice: returns (Time_Frames, Freq_Bins_31, Channels)
        # Convert to float PyTorch tensor
        x = torch.from_numpy(np.array(self.features[idx])).float()
        
        if self.labels is not None:
            y = self.labels[idx]
            return x, y
        return x


# --- Example Execution Blueprint ---
if __name__ == "__main__":
    print("Pre-computation Pipeline Script initialized.")
    print("This script is designed as a standalone utility.")
    
    # 1. Define dummy dimensions simulating the StealthyIMU dataset
    # (N, Time_Steps, Channels) where average duration ~4.06 seconds at 500Hz -> ~2030 Time_Steps
    N_train, N_val, N_test = 24554, 3069, 3070
    Time_Steps = 2030
    Channels = 6
    
    print(f"\nSimulating feature extraction for:")
    print(f"- Train Split: {N_train} samples")
    print(f"- Val Split: {N_val} samples")
    print(f"- Test Split: {N_test} samples")
    
    # Let's mock a single batch computation to demonstrate and test functionality
    dummy_raw_train = np.random.randn(10, Time_Steps, Channels)
    
    print("\nExtracting features for 10 mock raw samples...")
    specs = precompute_features(
        dummy_raw_train, 
        sample_rate=500, 
        n_fft=80, 
        win_length=80, 
        hop_length=20, 
        channel_axis=-1
    )
    
    print(f"Input shape: {dummy_raw_train.shape}")
    print(f"Output spectrogram shape: {specs.shape}")
    print("Expected shape representation: (N, Time_Frames, Freq_Bins, Channels)")
    print("Extraction successful! Saving and loading mock data to verify Dataset Class...")
    
    # Verify save & custom dataset loading
    output_dir = "./precomputed_output"
    os.makedirs(output_dir, exist_ok=True)
    temp_save_path = os.path.join(output_dir, "x_mock_spec.npy")
    np.save(temp_save_path, specs)
    
    # Instantiate custom dataset with memory mapping
    dataset = PrecomputedDataset(features_path=temp_save_path, use_mmap=True)
    sample_item = dataset[0]
    print(f"Dataset retrieved sample tensor shape: {sample_item.shape}")
    print("Verification passed successfully.")
