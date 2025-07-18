# tests/test_ecg_system.py
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import pandas as pd

class ECGSystemValidator:
    def __init__(self):
        self.test_results = {}
    
    def validate_qrs_detection(self, ecg_signal, true_r_peaks, detected_r_peaks, 
                              tolerance=0.05, sampling_rate=360):
        """
        Validate QRS detection accuracy
        """
        tolerance_samples = int(tolerance * sampling_rate)
        
        # Match detected peaks with true peaks
        true_positives = 0
        false_positives = 0
        
        matched_detected = []
        
        for detected_peak in detected_r_peaks:
            # Find closest true peak
            distances = np.abs(true_r_peaks - detected_peak)
            min_distance = np.min(distances)
            
            if min_distance <= tolerance_samples:
                true_positives += 1
                matched_detected.append(detected_peak)
            else:
                false_positives += 1
        
        false_negatives = len(true_r_peaks) - true_positives
        
        # Calculate metrics
        sensitivity = true_positives / (true_positives + false_negatives) if (true_positives + false_negatives) > 0 else 0
        specificity = true_positives / (true_positives + false_positives) if (true_positives + false_positives) > 0 else 0
        f1 = 2 * (sensitivity * specificity) / (sensitivity + specificity) if (sensitivity + specificity) > 0 else 0
        
        results = {
            'sensitivity': sensitivity,
            'specificity': specificity,
            'f1_score': f1,
            'true_positives': true_positives,
            'false_positives': false_positives,
            'false_negatives': false_negatives,
            'total_detected': len(detected_r_peaks),
            'total_true': len(true_r_peaks)
        }
        
        return results
    
    def validate_heart_rate_accuracy(self, calculated_hr, true_hr, tolerance=5):
        """
        Validate heart rate calculation accuracy
        """
        if len(calculated_hr) == 0 or len(true_hr) == 0:
            return {'accuracy': 0, 'mae': float('inf'), 'rmse': float('inf')}
        
        # Ensure same length
        min_len = min(len(calculated_hr), len(true_hr))
        calc_hr = calculated_hr[:min_len]
        true_hr = true_hr[:min_len]
        
        # Calculate accuracy within tolerance
        accurate_predictions = np.abs(calc_hr - true_hr) <= tolerance
        accuracy = np.mean(accurate_predictions)
        
        # Calculate error metrics
        mae = np.mean(np.abs(calc_hr - true_hr))
        rmse = np.sqrt(np.mean((calc_hr - true_hr)**2))
        
        results = {
            'accuracy': accuracy,
            'mae': mae,
            'rmse': rmse,
            'correlation': np.corrcoef(calc_hr, true_hr)[0, 1] if len(calc_hr) > 1 else 0
        }
        
        return results
    
    def test_preprocessing_effectiveness(self, original_signal, processed_signal):
        """
        Test preprocessing effectiveness
        """
        # Signal-to-noise ratio improvement
        original_power = np.mean(original_signal**2)
        processed_power = np.mean(processed_signal**2)
        
        # Noise estimation using high-frequency components
        original_noise = np.std(np.diff(original_signal))
        processed_noise = np.std(np.diff(processed_signal))
        
        snr_improvement = (processed_power / processed_noise**2) / (original_power / original_noise**2)
        
        # Baseline stability
        original_baseline_var = np.var(original_signal)
        processed_baseline_var = np.var(processed_signal)
        
        results = {
            'snr_improvement': snr_improvement,
            'noise_reduction': (original_noise - processed_noise) / original_noise,
            'baseline_stability': processed_baseline_var / original_baseline_var,
            'signal_preservation': np.corrcoef(original_signal, processed_signal)[0, 1]
        }
        
        return results
    
    def benchmark_processing_speed(self, ecg_processor, test_signals, num_iterations=10):
        """
        Benchmark processing speed
        """
        import time
        
        processing_times = []
        
        for _ in range(num_iterations):
            for signal in test_signals:
                start_time = time.time()
                
                # Process signal
                processed = ecg_processor.preprocess_signal(signal)
                qrs_results = ecg_processor.detect_qrs_with_metrics(processed)
                
                end_time = time.time()
                processing_times.append(end_time - start_time)
        
        avg_time = np.mean(processing_times)
        std_time = np.std(processing_times)
        
        # Calculate processing factor (how much faster than real-time)
        signal_duration = len(test_signals[0]) / 360  # Assume 360 Hz
        processing_factor = signal_duration / avg_time
        
        results = {
            'average_time': avg_time,
            'std_time': std_time,
            'processing_factor': processing_factor,
            'real_time_capable': processing_factor > 1
        }
        
        return results
    
    def comprehensive_system_test(self):
        """
        Run comprehensive system tests
        """
        from preprocessing import ECGDataLoader, ECGPreprocessor
        from detection import QRSDetector
        from features import ECGFeatureExtractor
        
        print("Running comprehensive ECG system tests...")
        
        # Initialize components
        loader = ECGDataLoader()
        preprocessor = ECGPreprocessor()
        detector = QRSDetector()
        extractor = ECGFeatureExtractor()
        
        test_results = {}
        
        # Test 1: Generate multiple test signals
        print("1. Generating test signals...")
        test_signals = []
        true_r_peaks_list = []
        
        for i in range(5):
            # Generate signals with different characteristics
            duration = 30
            noise_level = 0.02 + i * 0.01
            data = loader.generate_sample_ecg(duration=duration, noise_level=noise_level)
            test_signals.append(data['signal'])
            
            # Generate true R-peaks for validation (simplified)
            heart_rate = 70 + i * 5
            period = 60 / heart_rate
            true_peaks = np.arange(period, duration, period) * 360
            true_r_peaks_list.append(true_peaks.astype(int))
        
        # Test 2: Preprocessing validation
        print("2. Testing preprocessing...")
        preprocessing_results = []
        
        for signal in test_signals:
            processed_signal = preprocessor.preprocess_signal(signal)
            prep_result = self.test_preprocessing_effectiveness(signal, processed_signal)
            preprocessing_results.append(prep_result)
        
        test_results['preprocessing'] = {
            'avg_snr_improvement': np.mean([r['snr_improvement'] for r in preprocessing_results]),
            'avg_noise_reduction': np.mean([r['noise_reduction'] for r in preprocessing_results]),
            'avg_signal_preservation': np.mean([r['signal_preservation'] for r in preprocessing_results])
        }
        
        # Test 3: QRS Detection validation
        print("3. Testing QRS detection...")
        qrs_detection_results = []
        
        for i, signal in enumerate(test_signals):
            processed_signal = preprocessor.preprocess_signal(signal)
            qrs_results = detector.detect_qrs_with_metrics(processed_signal)
            
            # Validate against true peaks
            validation = self.validate_qrs_detection(
                processed_signal, 
                true_r_peaks_list[i], 
                qrs_results['r_peaks']
            )
            qrs_detection_results.append(validation)
        
        test_results['qrs_detection'] = {
            'avg_sensitivity': np.mean([r['sensitivity'] for r in qrs_detection_results]),
            'avg_specificity': np.mean([r['specificity'] for r in qrs_detection_results]),
            'avg_f1_score': np.mean([r['f1_score'] for r in qrs_detection_results])
        }
        
        # Test 4: Processing speed benchmark
        print("4. Benchmarking processing speed...")
        speed_results = self.benchmark_processing_speed(
            type('ECGProcessor', (), {
                'preprocess_signal': preprocessor.preprocess_signal,
                'detect_qrs_with_metrics': detector.detect_qrs_with_metrics
            })(),
            test_signals[:3]  # Use first 3 signals for speed test
        )
        
        test_results['performance'] = speed_results
        
        # Test 5: Feature extraction validation
        print("5. Testing feature extraction...")
        feature_results = []
        
        for signal in test_signals:
            processed_signal = preprocessor.preprocess_signal(signal)
            qrs_results = detector.detect_qrs_with_metrics(processed_signal)
            features = extractor.extract_all_features(
                processed_signal, 
                qrs_results['r_peaks'], 
                qrs_results['rr_intervals']
            )
            feature_results.append(features)
        
        # Check feature consistency
        feature_names = feature_results[0].keys()
        feature_stability = {}
        
        for feature_name in feature_names:
            if isinstance(feature_results[0][feature_name], (int, float)):
                values = [r[feature_name] for r in feature_results if feature_name in r]
                if values:
                    feature_stability[feature_name] = {
                        'mean': np.mean(values),
                        'std': np.std(values),
                        'cv': np.std(values) / np.mean(values) if np.mean(values) != 0 else 0
                    }
        
        test_results['features'] = feature_stability
        
        return test_results
    
    def generate_test_report(self, test_results):
        """
        Generate comprehensive test report
        """
        report = """
# ECG Processing System Test Report

## Overview
This report contains the results of comprehensive testing of the ECG processing system.

## Test Results Summary

### 1. Preprocessing Performance
"""
        
        prep = test_results.get('preprocessing', {})
        report += f"""
- **SNR Improvement**: {prep.get('avg_snr_improvement', 0):.2f}x
- **Noise Reduction**: {prep.get('avg_noise_reduction', 0)*100:.1f}%
- **Signal Preservation**: {prep.get('avg_signal_preservation', 0):.3f}
"""
        
        qrs = test_results.get('qrs_detection', {})
        report += f"""
### 2. QRS Detection Accuracy
- **Sensitivity**: {qrs.get('avg_sensitivity', 0)*100:.1f}%
- **Specificity**: {qrs.get('avg_specificity', 0)*100:.1f}%
- **F1 Score**: {qrs.get('avg_f1_score', 0):.3f}
"""
        
        perf = test_results.get('performance', {})
        report += f"""
### 3. Processing Performance
- **Average Processing Time**: {perf.get('average_time', 0):.3f} seconds
- **Processing Factor**: {perf.get('processing_factor', 0):.1f}x real-time
- **Real-time Capable**: {'Yes' if perf.get('real_time_capable', False) else 'No'}
"""
        
        features = test_results.get('features', {})
        report += f"""
### 4. Feature Extraction Stability
"""
        for feature_name, stats in features.items():
            if isinstance(stats, dict):
                report += f"- **{feature_name}**: {stats['mean']:.2f} ± {stats['std']:.2f} (CV: {stats['cv']:.2f})\n"
        
        report += """
## Recommendations

### Performance
"""
        
        # Add recommendations based on results
        if qrs.get('avg_sensitivity', 0) < 0.95:
            report += "- Consider tuning QRS detection parameters to improve sensitivity\n"
        
        if perf.get('processing_factor', 0) < 1:
            report += "- Optimize processing algorithms for real-time performance\n"
        
        if prep.get('avg_snr_improvement', 0) < 2:
            report += "- Review preprocessing filters for better noise reduction\n"
        
        report += """
### Clinical Usage
- Validate with clinical data before medical use
- Implement additional safety checks for critical applications
- Consider regulatory requirements for medical devices

## Conclusion
"""
        
        overall_score = (
            qrs.get('avg_f1_score', 0) * 0.4 +
            prep.get('avg_signal_preservation', 0) * 0.3 +
            min(perf.get('processing_factor', 0), 1) * 0.3
        )
        
        report += f"Overall system performance score: {overall_score:.2f}/1.0"
        
        if overall_score > 0.8:
            report += "\n\nThe system demonstrates excellent performance and is ready for advanced testing."
        elif overall_score > 0.6:
            report += "\n\nThe system shows good performance with room for optimization."
        else:
            report += "\n\nThe system requires significant improvements before deployment."
        
        return report

# Example usage and main execution script
def main():
    """
    Main execution script for ECG processing system
    """
    print("ECG Signal Processing System")
    print("=" * 50)
    
    # Import all components
    from preprocessing import ECGDataLoader, ECGPreprocessor
    from detection import QRSDetector, ArrhythmiaDetector
    from features import ECGFeatureExtractor
    from visualization import ECGVisualizer
    
    # Initialize system components
    loader = ECGDataLoader()
    preprocessor = ECGPreprocessor()
    detector = QRSDetector()
    arrhythmia_detector = ArrhythmiaDetector()
    extractor = ECGFeatureExtractor()
    visualizer = ECGVisualizer()
    
    print("1. Loading ECG data...")
    # You can replace this with real data loading
    data = loader.generate_sample_ecg(duration=60, noise_level=0.03)
    print(f"   Loaded {len(data['signal'])} samples ({len(data['signal'])/360:.1f} seconds)")
    
    print("2. Preprocessing signal...")
    clean_signal = preprocessor.preprocess_signal(data['signal'])
    quality = preprocessor.signal_quality_assessment(clean_signal)
    print(f"   Signal quality: {quality['quality']} (SNR: {quality['snr']:.1f} dB)")
    
    print("3. Detecting QRS complexes...")
    qrs_results = detector.detect_qrs_with_metrics(clean_signal)
    print(f"   Detected {qrs_results['num_beats']} heartbeats")
    print(f"   Average heart rate: {qrs_results['average_heart_rate']:.1f} bpm")
    
    print("4. Extracting features...")
    features = extractor.extract_all_features(
        clean_signal, 
        qrs_results['r_peaks'], 
        qrs_results['rr_intervals']
    )
    print(f"   Extracted {len([k for k, v in features.items() if not isinstance(v, np.ndarray)])} numerical features")
    
    print("5. Analyzing for abnormalities...")
    analysis_results = {**qrs_results, 'features': features}
    report = arrhythmia_detector.generate_report(clean_signal, analysis_results)
    print(f"   Rhythm classification: {report['rhythm_classification']}")
    
    print("6. Creating visualizations...")
    # Main analysis plot
    fig = visualizer.plot_ecg_analysis(clean_signal, analysis_results, data['time'])
    plt.savefig('ecg_analysis.png', dpi=300, bbox_inches='tight')
    print("   Saved analysis plot as 'ecg_analysis.png'")
    
    # HRV analysis
    hrv_fig = visualizer.plot_hrv_analysis(qrs_results['rr_intervals'], features)
    plt.savefig('hrv_analysis.png', dpi=300, bbox_inches='tight')
    print("   Saved HRV analysis as 'hrv_analysis.png'")
    
    print("7. Generating report...")
    print("\n" + "="*50)
    print("ANALYSIS REPORT")
    print("="*50)
    print(f"Patient ID: DEMO_001")
    print(f"Recording duration: {len(data['signal'])/360:.1f} seconds")
    print(f"Total heartbeats: {report['num_beats']}")
    print(f"Heart rate: {report['heart_rate']['mean']:.1f} ± {report['heart_rate']['std']:.1f} bpm")
    print(f"Rhythm: {report['rhythm_classification']}")
    print(f"RMSSD: {report['hrv_analysis']['rmssd']:.1f} ms")
    print(f"QRS width: {report['morphology']['qrs_width']:.1f} ms")
    
    if report['abnormalities']:
        print("\nABNORMALITIES DETECTED:")
        for abnormality in report['abnormalities']:
            print(f"  ⚠️  {abnormality}")
    else:
        print("\n✅ No abnormalities detected")
    
    print("\n" + "="*50)
    print("Analysis complete! Check the generated plots for detailed visualization.")
    
    return {
        'signal_data': data,
        'processed_signal': clean_signal,
        'analysis_results': analysis_results,
        'report': report,
        'features': features
    }

if __name__ == "__main__":
    # Run main analysis
    results = main()
    
    # Optional: Run comprehensive tests
    print("\nRunning system validation tests...")
    validator = ECGSystemValidator()
    test_results = validator.comprehensive_system_test()
    
    # Generate test report
    test_report = validator.generate_test_report(test_results)
    
    # Save test report
    with open('test_report.md', 'w') as f:
        f.write(test_report)
    
    print("Test report saved as 'test_report.md'")
    print("\nSystem validation complete!")