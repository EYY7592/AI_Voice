import numpy as np
import librosa
import parselmouth
from parselmouth.praat import call
import torch

def test_feature_extraction():
    sr = 16000
    duration = 1.0
    # Create a 1s sine wave + noise
    t = np.linspace(0, duration, int(sr * duration))
    audio = 0.5 * np.sin(2 * np.pi * 440 * t) + 0.1 * np.random.randn(len(t))
    
    # Test function logic
    print("Testing with numpy array...")
    try:
        # User's function logic (simplified for check)
        snd = parselmouth.Sound(audio, sampling_frequency=sr)
        pitch = call(snd, 'To Pitch', 0.0, 75, 600)
        print("Pitch extraction OK")
        
        point_process = call(snd, 'To PointProcess (periodic, cc)', 75, 600)
        print("PointProcess OK")
        
        jitter = call(point_process, 'Get jitter (local)', 0, 0, 0.0001, 0.02, 1.3)
        shimmer = call([snd, point_process], 'Get shimmer (local)', 0, 0, 0.0001, 0.02, 1.3, 1.6)
        print(f"Jitter: {jitter}, Shimmer: {shimmer}")
        
    except Exception as e:
        print(f"Failed: {e}")

    # Test with tensor
    print("\nTesting with torch tensor...")
    audio_tensor = torch.from_numpy(audio)
    try:
        snd = parselmouth.Sound(audio_tensor, sampling_frequency=sr)
        print("Tensor support OK")
    except Exception as e:
        print(f"Tensor failed (Expected): {e}")

if __name__ == "__main__":
    test_feature_extraction()
