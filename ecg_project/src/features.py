# src/features.py
import numpy as np
from scipy import signal, stats
import matplotlib.pyplot as plt

class ECGFeatureExtractor:
    def __init__(self, sampling_rate=360):
        self.fs = sampling_rate
    
    def calculate_hrv_features(self, rr_intervals):
        """
        Calculate Heart Rate Variability features
        """
        if len(rr_intervals) < 2:
            return {}
        
        # Time domain features
        rr_ms = rr_intervals * 1000  # Convert to milliseconds
        
        # Statistical measures
        mean_rr = np.mean(rr_ms)
        std_rr = np.std(rr_ms)
        rmssd = np.sqrt(np.mean(np.diff(rr_ms)**2))
        
        # pNN50: percentage of RR intervals that differ by more than 50ms
        diff_rr = np.abs(np.diff(rr_ms))
        pnn50 = (np.sum(diff_rr > 50) / len(diff_rr)) * 100
        
        # Triangular index
        hist, bins = np.histogram(rr_ms, bins=50)
        tri_index = len(rr_ms) / np.max(hist)
        
        # Frequency domain features (simplified)
        # Calculate power spectral density
        if len(rr_intervals) > 10:
            freqs, psd = signal.welch(rr_intervals, fs=1/np.mean(rr_intervals))
            
            # Define frequency bands
            lf_band = (freqs >= 0.04) & (freqs <= 0.15)  # Low frequency
            hf_band = (freqs >= 0.15) & (freqs <= 0.4)   # High frequency
            
            lf_power = np.trapz(psd[lf_band], freqs[lf_band])
            hf_power = np.trapz(psd[hf_band], freqs[hf_band])
            lf_hf_ratio = lf_power / hf_power if hf_power > 0 else 0
        else:
            lf_power = hf_power = lf_hf_ratio = 0
        
        hrv_features = {
            'mean_rr': mean_rr,
            'std_rr': std_rr,
            'rmssd': rmssd,
            'pnn50': pnn50,
            'triangular_index': tri_index,
            'lf_power': lf_power,
            'hf_power': hf_power,
            'lf_hf_ratio': lf_hf_ratio
        }
        
        return hrv_features
    
    def extract_qrs_morphology(self, ecg_signal, r_peaks, window_size=0.2):
        """
        Extract QRS morphological features
        """
        window_samples = int(window_size * self.fs)
        qrs_complexes = []
        
        for r_peak in r_peaks:
            # Define window around R-peak
            start = max(0, r_peak - window_samples // 2)
            end = min(len(ecg_signal), r_peak + window_samples // 2)
            
            qrs_segment = ecg_signal[start:end]
            
            if len(qrs_segment) == window_samples:
                qrs_complexes.append(qrs_segment)
        
        if not qrs_complexes:
            return {}
        
        qrs_complexes = np.array(qrs_complexes)
        
        # Calculate morphological features
        features = {
            'qrs_width': self.calculate_qrs_width(qrs_complexes),
            'qrs_amplitude': np.mean([np.max(qrs) - np.min(qrs) for qrs in qrs_complexes]),
            'qrs_area': np.mean([np.trapz(np.abs(qrs)) for qrs in qrs_complexes]),
            'qrs_slope': self.calculate_qrs_slope(qrs_complexes),
            'qrs_template': np.mean(qrs_complexes, axis=0)  # Average QRS template
        }
        
        return features
    
    def calculate_qrs_width(self, qrs_complexes):
        """Calculate average QRS width"""
        widths = []
        for qrs in qrs_complexes:
            # Find where QRS starts and ends (simplified)
            threshold = 0.1 * np.max(np.abs(qrs))
            above_threshold = np.abs(qrs) > threshold
            
            if np.any(above_threshold):
                start_idx = np.where(above_threshold)[0][0]
                end_idx = np.where(above_threshold)[0][-1]
                width = (end_idx - start_idx) / self.fs * 1000  # Convert to ms
                widths.append(width)
        
        return np.mean(widths) if widths else 0
    
    def calculate_qrs_slope(self, qrs_complexes):
        """Calculate average QRS slope"""
        slopes = []
        for qrs in qrs_complexes:
            # Find R-peak
            r_idx = np.argmax(np.abs(qrs))
            
            # Calculate slope before and after R-peak
            if r_idx > 5 and r_idx < len(qrs) - 5:
                upslope = (qrs[r_idx] - qrs[r_idx-5]) / 5
                downslope = (qrs[r_idx+5] - qrs[r_idx]) / 5
                slopes.append(abs(upslope) + abs(downslope))
        
        return np.mean(slopes) if slopes else 0
    
    def extract_all_features(self, ecg_signal, r_peaks, rr_intervals):
        """
        Extract all ECG features
        """
        features = {}
        
        # HRV features
        features.update(self.calculate_hrv_features(rr_intervals))
        
        # QRS morphology features
        features.update(self.extract_qrs_morphology(ecg_signal, r_peaks))
        
        # Additional temporal features
        features.update({
            'heart_rate_mean': 60 / np.mean(rr_intervals) if len(rr_intervals) > 0 else 0,
            'heart_rate_std': np.std(60 / rr_intervals) if len(rr_intervals) > 0 else 0,
            'num_beats': len(r_peaks)
        })
        
        return features

# Test feature extraction
if __name__ == "__main__":
    from preprocessing import ECGDataLoader, ECGPreprocessor
    from detection import QRSDetector
    
    # Load and process data
    loader = ECGDataLoader()
    preprocessor = ECGPreprocessor()
    detector = QRSDetector()
    extractor = ECGFeatureExtractor()
    
    # Generate test signal
    data = loader.generate_sample_ecg(duration=30, noise_level=0.02)
    clean_signal = preprocessor.preprocess_signal(data['signal'])
    
    # Detect QRS
    qrs_results = detector.detect_qrs_with_metrics(clean_signal)
    
    # Extract features
    features = extractor.extract_all_features(
        clean_signal, 
        qrs_results['r_peaks'], 
        qrs_results['rr_intervals']
    )
    
    print("Extracted ECG Features:")
    for key, value in features.items():
        if isinstance(value, np.ndarray):
            print(f"{key}: array of shape {value.shape}")
        else:
            print(f"{key}: {value:.3f}")

# src/detection.py (continued)

class ArrhythmiaDetector:
    def __init__(self, sampling_rate=360):
        self.fs = sampling_rate
    
    def detect_bradycardia(self, heart_rate, threshold=60):
        """Detect bradycardia (slow heart rate)"""
        return np.mean(heart_rate) < threshold
    
    def detect_tachycardia(self, heart_rate, threshold=100):
        """Detect tachycardia (fast heart rate)"""
        return np.mean(heart_rate) > threshold
    
    def detect_arrhythmia(self, rr_intervals, threshold=0.2):
        """Detect irregular rhythm based on RR interval variability"""
        if len(rr_intervals) < 3:
            return False
        
        # Calculate coefficient of variation
        cv = np.std(rr_intervals) / np.mean(rr_intervals)
        return cv > threshold
    
    def detect_atrial_fibrillation(self, rr_intervals, features):
        """
        Simplified atrial fibrillation detection
        Based on RR interval irregularity and HRV features
        """
        if len(rr_intervals) < 5:
            return False
        
        # Check for irregular RR intervals
        irregular_rhythm = self.detect_arrhythmia(rr_intervals, threshold=0.15)
        
        # Check HRV features
        high_variability = features.get('rmssd', 0) > 50  # ms
        
        return irregular_rhythm and high_variability
    
    def detect_premature_beats(self, ecg_signal, r_peaks, qrs_features):
        """
        Detect premature ventricular contractions (PVCs)
        """
        if len(r_peaks) < 3:
            return []
        
        premature_beats = []
        rr_intervals = np.diff(r_peaks) / self.fs
        
        for i in range(1, len(rr_intervals)):
            current_rr = rr_intervals[i]
            prev_rr = rr_intervals[i-1]
            
            # Check for early beat (shorter RR interval)
            if current_rr < 0.8 * prev_rr:
                # Check for compensatory pause
                if i < len(rr_intervals) - 1:
                    next_rr = rr_intervals[i+1]
                    if next_rr > 1.2 * prev_rr:
                        premature_beats.append(r_peaks[i+1])
        
        return premature_beats
    
    def classify_rhythm(self, ecg_signal, r_peaks, rr_intervals, features):
        """
        Classify the cardiac rhythm
        """
        if len(r_peaks) < 3:
            return "Insufficient data"
        
        heart_rate = 60 / np.mean(rr_intervals)
        
        # Check for different conditions
        conditions = []
        
        if self.detect_bradycardia([heart_rate]):
            conditions.append("Bradycardia")
        
        if self.detect_tachycardia([heart_rate]):
            conditions.append("Tachycardia")
        
        if self.detect_atrial_fibrillation(rr_intervals, features):
            conditions.append("Atrial Fibrillation")
        
        if self.detect_arrhythmia(rr_intervals):
            conditions.append("Irregular Rhythm")
        
        premature_beats = self.detect_premature_beats(ecg_signal, r_peaks, features)
        if len(premature_beats) > 0:
            conditions.append(f"Premature Beats ({len(premature_beats)})")
        
        if not conditions:
            return "Normal Sinus Rhythm"
        
        return ", ".join(conditions)
    
    def generate_report(self, ecg_signal, analysis_results):
        """
        Generate a comprehensive analysis report
        """
        r_peaks = analysis_results['r_peaks']
        rr_intervals = analysis_results['rr_intervals']
        features = analysis_results['features']
        
        report = {
            'timestamp': np.datetime64('now'),
            'signal_length': len(ecg_signal) / self.fs,
            'num_beats': len(r_peaks),
            'heart_rate': {
                'mean': np.mean(analysis_results['heart_rate_instant']),
                'std': np.std(analysis_results['heart_rate_instant']),
                'min': np.min(analysis_results['heart_rate_instant']),
                'max': np.max(analysis_results['heart_rate_instant'])
            },
            'rhythm_classification': self.classify_rhythm(ecg_signal, r_peaks, rr_intervals, features),
            'hrv_analysis': {
                'rmssd': features.get('rmssd', 0),
                'pnn50': features.get('pnn50', 0),
                'lf_hf_ratio': features.get('lf_hf_ratio', 0)
            },
            'morphology': {
                'qrs_width': features.get('qrs_width', 0),
                'qrs_amplitude': features.get('qrs_amplitude', 0)
            },
            'abnormalities': []
        }
        
        # Add detected abnormalities
        if self.detect_bradycardia([report['heart_rate']['mean']]):
            report['abnormalities'].append("Bradycardia detected")
        
        if self.detect_tachycardia([report['heart_rate']['mean']]):
            report['abnormalities'].append("Tachycardia detected")
        
        return report

# Test abnormality detection
if __name__ == "__main__":
    from preprocessing import ECGDataLoader, ECGPreprocessor
    from detection import QRSDetector
    from features import ECGFeatureExtractor
    
    # Initialize classes
    loader = ECGDataLoader()
    preprocessor = ECGPreprocessor()
    detector = QRSDetector()
    extractor = ECGFeatureExtractor()
    arrhythmia_detector = ArrhythmiaDetector()
    
    # Generate test signal with some irregularity
    data = loader.generate_sample_ecg(duration=30, noise_level=0.02)
    clean_signal = preprocessor.preprocess_signal(data['signal'])
    
    # Detect QRS
    qrs_results = detector.detect_qrs_with_metrics(clean_signal)
    
    # Extract features
    features = extractor.extract_all_features(
        clean_signal, 
        qrs_results['r_peaks'], 
        qrs_results['rr_intervals']
    )
    
    # Combine results
    analysis_results = {**qrs_results, 'features': features}
    
    # Generate report
    report = arrhythmia_detector.generate_report(clean_signal, analysis_results)
    
    print("ECG Analysis Report:")
    print(f"Signal duration: {report['signal_length']:.1f} seconds")
    print(f"Number of beats: {report['num_beats']}")
    print(f"Heart rate: {report['heart_rate']['mean']:.1f} ± {report['heart_rate']['std']:.1f} bpm")
    print(f"Rhythm classification: {report['rhythm_classification']}")
    print(f"RMSSD: {report['hrv_analysis']['rmssd']:.1f} ms")
    print(f"QRS width: {report['morphology']['qrs_width']:.1f} ms")
    
    if report['abnormalities']:
        print("Detected abnormalities:")
        for abnormality in report['abnormalities']:
            print(f"  - {abnormality}")
    else:
        print("No abnormalities detected")