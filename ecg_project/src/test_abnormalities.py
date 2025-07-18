#!/usr/bin/env python3
import sys
sys.path.append('src')

from preprocessing import ECGDataLoader, ECGPreprocessor
from detection import QRSDetector
from features import ECGFeatureExtractor

# Simple test with modified thresholds
def quick_test():
    loader = ECGDataLoader()
    preprocessor = ECGPreprocessor()
    detector = QRSDetector()
    extractor = ECGFeatureExtractor()
    
    # Test 1: Bradycardia
    print("🔍 Testing Bradycardia...")
    data = loader.generate_sample_ecg(duration=30, noise_level=0.02)
    
    # Modify the signal to simulate bradycardia
    # Reduce heart rate by stretching the signal
    import numpy as np
    from scipy import signal as sp_signal
    
    # Resample to simulate slower heart rate
    stretched_signal = sp_signal.resample(data['signal'], int(len(data['signal']) * 1.4))
    data['signal'] = stretched_signal[:len(data['signal'])]  # Keep same length
    
    # Process
    clean_signal = preprocessor.preprocess_signal(data['signal'])
    qrs_results = detector.detect_qrs_with_metrics(clean_signal)
    
    print(f"Detected HR: {qrs_results['average_heart_rate']:.1f} bpm")
    print(f"Detected beats: {qrs_results['num_beats']}")
    
    # Manual abnormality check with lower threshold
    if qrs_results['average_heart_rate'] < 65:  # More sensitive
        print("🚨 BRADYCARDIA DETECTED!")
    else:
        print("✅ Normal rhythm")

if __name__ == "__main__":
    quick_test()