#!/usr/bin/env python3
"""
ECG Signal Processing System - Main Execution Script
===================================================

This is the main entry point for the ECG processing system.
Run this script to perform complete ECG analysis with visualization and reporting.

Usage:
    python main.py                           # Demo mode with synthetic ECG
    python main.py --mode real --record 100  # Real ECG from MIT-BIH database
    python main.py --test-abnormalities      # Test abnormality detection
    python main.py --heart-rate 50           # Test bradycardia
    python main.py --heart-rate 120          # Test tachycardia

Author: ECG Processing System
Version: 2.0 (with Real ECG Support)
"""

import numpy as np
import matplotlib.pyplot as plt
import os
import sys
import argparse
from datetime import datetime

# Add src directory to path for imports
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

# Import all ECG processing components
try:
    from preprocessing import ECGDataLoader, ECGPreprocessor, RealECGDataLoader
    from detection import QRSDetector, ArrhythmiaDetector
    from features import ECGFeatureExtractor
    from visualization import ECGVisualizer
    from realtime import RealTimeECGProcessor
except ImportError as e:
    print(f"Error importing modules: {e}")
    print("Make sure all required files are in the src/ directory")
    sys.exit(1)

def setup_directories():
    """Create necessary directories if they don't exist"""
    directories = ['data/raw', 'data/processed', 'data/samples', 'data/real_ecg', 'results', 'notebooks', 'tests']
    
    for directory in directories:
        os.makedirs(directory, exist_ok=True)
        print(f"✓ Directory '{directory}' ready")

def analyze_ecg_file(filepath, output_dir='results'):
    """
    Analyze ECG data from a file
    
    Args:
        filepath (str): Path to ECG data file (CSV format)
        output_dir (str): Directory to save results
    
    Returns:
        dict: Analysis results
    """
    print(f"\n📁 Loading ECG data from: {filepath}")
    
    # Initialize components
    loader = ECGDataLoader()
    preprocessor = ECGPreprocessor()
    detector = QRSDetector()
    arrhythmia_detector = ArrhythmiaDetector()
    extractor = ECGFeatureExtractor()
    visualizer = ECGVisualizer()
    
    # Load data
    data = loader.load_csv_data(filepath)
    if data is None:
        print("❌ Failed to load data")
        return None
    
    return process_ecg_data(data, output_dir, visualizer, preprocessor, 
                          detector, arrhythmia_detector, extractor)

def analyze_mitbih_record(record_name, output_dir='results'):
    """
    Analyze ECG data from MIT-BIH database (legacy method)
    
    Args:
        record_name (str): MIT-BIH record name (e.g., '100')
        output_dir (str): Directory to save results
    
    Returns:
        dict: Analysis results
    """
    print(f"\n🏥 Loading MIT-BIH record: {record_name}")
    
    # Initialize components
    loader = ECGDataLoader()
    preprocessor = ECGPreprocessor()
    detector = QRSDetector()
    arrhythmia_detector = ArrhythmiaDetector()
    extractor = ECGFeatureExtractor()
    visualizer = ECGVisualizer()
    
    # Load data
    data = loader.load_mitbih_record(record_name)
    if data is None:
        print("❌ Failed to load MIT-BIH record")
        return None
    
    return process_ecg_data(data, output_dir, visualizer, preprocessor, 
                          detector, arrhythmia_detector, extractor)

def analyze_real_ecg_data(record_name, database='mitdb', output_dir='results'):
    """
    Analyze real ECG data from medical databases with expert comparison
    
    Args:
        record_name (str): Record name (e.g., '100', '101')
        database (str): Database name ('mitdb', 'afdb', etc.)
        output_dir (str): Directory to save results
    
    Returns:
        dict: Analysis results with expert annotations
    """
    print(f"\n🏥 Loading real ECG data from {database} record {record_name}")
    
    # Initialize components
    real_loader = RealECGDataLoader()
    preprocessor = ECGPreprocessor()
    detector = QRSDetector()
    arrhythmia_detector = ArrhythmiaDetector()
    extractor = ECGFeatureExtractor()
    visualizer = ECGVisualizer()
    
    # Get record information
    record_info = real_loader.get_record_info(record_name, database)
    print(f"📋 Record info: {record_info['description']}")
    
    # Load real ECG data
    data = real_loader.load_mitbih_record(record_name, database)
    if data is None:
        print("❌ Failed to load real ECG data")
        return None
    
    print(f"📊 Loaded {len(data['signal'])} samples ({data['duration']:.1f} seconds)")
    print(f"🏷️  Found {len(data['annotations'])} expert annotations")
    
    # Process the data using existing pipeline
    results = process_ecg_data(data, output_dir, visualizer, preprocessor, 
                              detector, arrhythmia_detector, extractor)
    
    # Add expert comparison
    if results and data['annotations']:
        print("\n👨‍⚕️ Comparing with expert annotations...")
        
        # Count abnormal beats according to experts
        abnormal_beats = sum(1 for ann in data['annotations'] if ann['is_abnormal'])
        total_beats = len(data['annotations'])
        abnormal_percentage = (abnormal_beats / total_beats) * 100
        
        print(f"   Expert labels: {abnormal_beats}/{total_beats} abnormal beats ({abnormal_percentage:.1f}%)")
        
        # Show annotation types
        annotation_types = {}
        for ann in data['annotations']:
            desc = ann['description']
            annotation_types[desc] = annotation_types.get(desc, 0) + 1
        
        print("   Expert annotations found:")
        for ann_type, count in sorted(annotation_types.items()):
            if count > 0:  # Only show types that exist
                print(f"     • {ann_type}: {count} beats")
        
        # Add expert data to results
        results['expert_annotations'] = data['annotations']
        results['expert_summary'] = {
            'abnormal_beats': abnormal_beats,
            'total_beats': total_beats,
            'abnormal_percentage': abnormal_percentage,
            'annotation_types': annotation_types
        }
        results['record_info'] = record_info
    
    return results

def analyze_synthetic_ecg(duration=60, noise_level=0.03, output_dir='results', heart_rate=75):
    """
    Analyze synthetic ECG data (for demonstration)
    
    Args:
        duration (float): Duration in seconds
        noise_level (float): Noise level (0-1)
        output_dir (str): Directory to save results
        heart_rate (int): Target heart rate for synthetic ECG
    
    Returns:
        dict: Analysis results
    """
    print(f"\n🧪 Generating synthetic ECG data ({duration}s, noise={noise_level:.3f}, HR={heart_rate})")
    
    # Initialize components
    loader = ECGDataLoader()
    preprocessor = ECGPreprocessor()
    detector = QRSDetector()
    arrhythmia_detector = ArrhythmiaDetector()
    extractor = ECGFeatureExtractor()
    visualizer = ECGVisualizer()
    
    # Generate synthetic data with specified heart rate
    data = loader.generate_sample_ecg(duration=duration, noise_level=noise_level, heart_rate=heart_rate)
    
    return process_ecg_data(data, output_dir, visualizer, preprocessor, 
                          detector, arrhythmia_detector, extractor)

def process_ecg_data(data, output_dir, visualizer, preprocessor, detector, 
                    arrhythmia_detector, extractor):
    """
    Core ECG processing pipeline
    
    Args:
        data (dict): ECG data dictionary
        output_dir (str): Output directory
        visualizer, preprocessor, detector, arrhythmia_detector, extractor: Processing components
    
    Returns:
        dict: Complete analysis results
    """
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    print(f"   📊 Signal length: {len(data['signal'])} samples")
    print(f"   ⏱️  Duration: {len(data['signal'])/data['sampling_rate']:.1f} seconds")
    
    # Step 1: Preprocessing
    print("\n🔧 Preprocessing signal...")
    clean_signal = preprocessor.preprocess_signal(data['signal'])
    quality = preprocessor.signal_quality_assessment(clean_signal)
    print(f"   ✓ Signal quality: {quality['quality']} (SNR: {quality['snr']:.1f} dB)")
    
    # Step 2: QRS Detection
    print("\n💓 Detecting QRS complexes...")
    qrs_results = detector.detect_qrs_with_metrics(clean_signal)
    print(f"   ✓ Detected {qrs_results['num_beats']} heartbeats")
    print(f"   ✓ Average heart rate: {qrs_results['average_heart_rate']:.1f} bpm")
    
    # Step 3: Feature Extraction
    print("\n📈 Extracting features...")
    features = extractor.extract_all_features(
        clean_signal, 
        qrs_results['r_peaks'], 
        qrs_results['rr_intervals']
    )
    numerical_features = [k for k, v in features.items() if not isinstance(v, np.ndarray)]
    print(f"   ✓ Extracted {len(numerical_features)} numerical features")
    
    # Step 4: Abnormality Detection
    print("\n🔍 Analyzing for abnormalities...")
    analysis_results = {**qrs_results, 'features': features}
    report = arrhythmia_detector.generate_report(clean_signal, analysis_results)
    print(f"   ✓ Rhythm classification: {report['rhythm_classification']}")
    
    # Step 5: Visualization
    print("\n📊 Creating visualizations...")
    
    # Create time vector
    time_vector = data.get('time', np.arange(len(data['signal'])) / data['sampling_rate'])
    
    # Main analysis plot
    fig1 = visualizer.plot_ecg_analysis(clean_signal, analysis_results, time_vector)
    analysis_plot_path = os.path.join(output_dir, f'ecg_analysis_{timestamp}.png')
    plt.figure(fig1.number)
    plt.savefig(analysis_plot_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"   ✓ Saved analysis plot: {analysis_plot_path}")
    
    # HRV analysis plot
    fig2 = visualizer.plot_hrv_analysis(qrs_results['rr_intervals'], features)
    hrv_plot_path = os.path.join(output_dir, f'hrv_analysis_{timestamp}.png')
    plt.figure(fig2.number)
    plt.savefig(hrv_plot_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"   ✓ Saved HRV analysis: {hrv_plot_path}")
    
    # QRS morphology plot (if template available)
    if 'qrs_template' in features:
        fig3 = visualizer.plot_qrs_morphology(features['qrs_template'])
        qrs_plot_path = os.path.join(output_dir, f'qrs_morphology_{timestamp}.png')
        plt.figure(fig3.number)
        plt.savefig(qrs_plot_path, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"   ✓ Saved QRS morphology: {qrs_plot_path}")
    
    # Step 6: Generate Report
    print("\n📋 Generating comprehensive report...")
    
    report_content = generate_clinical_report(report, quality, features, timestamp, data)
    report_path = os.path.join(output_dir, f'ecg_report_{timestamp}.txt')
    
    with open(report_path, 'w') as f:
        f.write(report_content)
    
    print(f"   ✓ Saved clinical report: {report_path}")
    
    # Print summary to console
    print_summary_report(report, quality, features, data)
    
    # Return complete results
    return {
        'signal_data': data,
        'processed_signal': clean_signal,
        'analysis_results': analysis_results,
        'report': report,
        'features': features,
        'quality': quality,
        'files': {
            'analysis_plot': analysis_plot_path,
            'hrv_plot': hrv_plot_path,
            'report': report_path
        }
    }

def generate_clinical_report(report, quality, features, timestamp, data):
    """Generate detailed clinical report"""
    
    # Determine data source
    data_source = "Unknown"
    if 'record_name' in data:
        data_source = f"MIT-BIH Record {data['record_name']}"
    elif 'target_heart_rate' in data:
        data_source = f"Synthetic ECG (Target HR: {data['target_heart_rate']} bpm)"
    elif 'filepath' in data:
        data_source = f"CSV File: {data['filepath']}"
    
    report_content = f"""
ECG ANALYSIS REPORT
==================
Generated: {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}
Analysis ID: {timestamp}
Data Source: {data_source}

SIGNAL QUALITY ASSESSMENT
------------------------
Overall Quality: {quality['quality']}
Signal-to-Noise Ratio: {quality['snr']:.1f} dB
Clipping Detected: {'Yes' if quality['is_clipped'] else 'No'}
Flat Segments: {quality['flat_percentage']:.1f}%

RHYTHM ANALYSIS
--------------
Recording Duration: {report['signal_length']:.1f} seconds
Total Heartbeats: {report['num_beats']}
Rhythm Classification: {report['rhythm_classification']}

HEART RATE STATISTICS
--------------------
Mean Heart Rate: {report['heart_rate']['mean']:.1f} bpm
Standard Deviation: {report['heart_rate']['std']:.1f} bpm
Minimum Heart Rate: {report['heart_rate']['min']:.1f} bpm
Maximum Heart Rate: {report['heart_rate']['max']:.1f} bpm

HEART RATE VARIABILITY
---------------------
RMSSD: {report['hrv_analysis']['rmssd']:.1f} ms
pNN50: {report['hrv_analysis']['pnn50']:.1f}%
LF/HF Ratio: {report['hrv_analysis']['lf_hf_ratio']:.2f}

QRS MORPHOLOGY
-------------
Average QRS Width: {report['morphology']['qrs_width']:.1f} ms
Average QRS Amplitude: {report['morphology']['qrs_amplitude']:.3f}

CLINICAL FINDINGS
----------------
"""
    
    if report['abnormalities']:
        report_content += "ABNORMALITIES DETECTED:\n"
        for abnormality in report['abnormalities']:
            report_content += f"⚠️  {abnormality}\n"
    else:
        report_content += "✅ No significant abnormalities detected\n"
    
    report_content += f"""
RECOMMENDATIONS
--------------
1. Review signal quality if SNR < 10 dB
2. Consider clinical correlation for detected abnormalities
3. Repeat recording if signal quality is poor
4. Consult cardiologist for irregular rhythms

TECHNICAL NOTES
--------------
- Analysis performed using Pan-Tompkins QRS detection algorithm
- Heart rate variability calculated from RR intervals
- Preprocessing included baseline correction and filtering
- All measurements are automated estimates
- Abnormality detection thresholds: Bradycardia <65 bpm, Tachycardia >95 bpm

DISCLAIMER
----------
This analysis is for research/educational purposes only.
Clinical decisions should not be based solely on this automated analysis.
Always consult qualified medical professionals for clinical interpretation.
Real patient data should be validated against expert annotations.
"""
    
    return report_content

def print_summary_report(report, quality, features, data):
    """Print summary report to console"""
    
    print("\n" + "="*60)
    print("📊 ECG ANALYSIS SUMMARY")
    print("="*60)
    
    # Show data source
    if 'record_name' in data:
        print(f"📋 Record: {data['record_name']} from {data.get('database', 'unknown')} database")
    elif 'target_heart_rate' in data:
        print(f"🧪 Synthetic ECG (Target: {data['target_heart_rate']} bpm)")
    
    print(f"🔋 Signal Quality: {quality['quality']} (SNR: {quality['snr']:.1f} dB)")
    print(f"⏱️  Duration: {report['signal_length']:.1f} seconds")
    print(f"💓 Total Beats: {report['num_beats']}")
    print(f"📈 Heart Rate: {report['heart_rate']['mean']:.1f} ± {report['heart_rate']['std']:.1f} bpm")
    print(f"🎵 Rhythm: {report['rhythm_classification']}")
    print(f"📊 RMSSD: {report['hrv_analysis']['rmssd']:.1f} ms")
    print(f"⚡ QRS Width: {report['morphology']['qrs_width']:.1f} ms")
    
    if report['abnormalities']:
        print(f"\n⚠️  ABNORMALITIES DETECTED:")
        for abnormality in report['abnormalities']:
            print(f"   • {abnormality}")
    else:
        print(f"\n✅ No abnormalities detected")
    
    print("="*60)

def run_realtime_demo():
    """Run real-time ECG processing demonstration"""
    print("\n🔴 Starting Real-time ECG Processing Demo...")
    print("This will simulate 30 seconds of real-time ECG monitoring")
    print("Press Ctrl+C to stop early\n")
    
    try:
        processor = RealTimeECGProcessor()
        processor.simulate_realtime_data(duration=30)
        
        print("\n📊 Real-time Demo Results:")
        stats = processor.get_current_stats()
        print(f"   Final Heart Rate: {stats['heart_rate']:.1f} bpm")
        print(f"   Average Heart Rate: {stats['average_hr']:.1f} bpm")
        print(f"   Total Beats: {stats['total_beats']}")
        print(f"   Alerts Generated: {len(stats['recent_alerts'])}")
        
        if stats['recent_alerts']:
            print("   Recent Alerts:")
            for alert in stats['recent_alerts']:
                print(f"     • {alert['type']}: {alert['value']:.1f} ({alert['severity']})")
    
    except KeyboardInterrupt:
        print("\n⏹️  Real-time demo stopped by user")
    except Exception as e:
        print(f"\n❌ Error in real-time demo: {e}")

def test_abnormality_detection():
    """Test abnormality detection with various heart rate conditions"""
    print("\n🧪 Testing abnormality detection...")
    
    # Test cases with different heart rates
    test_cases = [
        (45, "severe bradycardia"),
        (55, "mild bradycardia"), 
        (110, "mild tachycardia"),
        (140, "severe tachycardia")
    ]
    
    for target_hr, condition in test_cases:
        print(f"\n🔍 Testing {condition} (target: {target_hr} bpm)")
        
        # Analyze synthetic ECG with target heart rate
        results = analyze_synthetic_ecg(duration=30, noise_level=0.02, output_dir='results', heart_rate=target_hr)
        
        if results:
            print(f"   Target HR: {target_hr} bpm")
            print(f"   Detected HR: {results['report']['heart_rate']['mean']:.1f} bpm")
            print(f"   Classification: {results['report']['rhythm_classification']}")
            if results['report']['abnormalities']:
                for abnormality in results['report']['abnormalities']:
                    print(f"   🚨 {abnormality}")
            else:
                print("   ✅ No abnormalities detected")
        else:
            print("   ❌ Analysis failed")

def list_available_records():
    """List available ECG records from databases"""
    print("\n📚 Available ECG Records:")
    print("=" * 50)
    
    real_loader = RealECGDataLoader()
    
    # MIT-BIH records
    mitdb_records = real_loader.get_available_records('mitdb')
    print(f"\n🏥 MIT-BIH Arrhythmia Database ({len(mitdb_records)} records):")
    print("   Common records with known conditions:")
    
    featured_records = [
        ('100', 'Normal sinus rhythm'),
        ('101', 'Atrial fibrillation'),
        ('106', 'Sinus rhythm with frequent PVCs'),
        ('108', 'Left bundle branch block'),
        ('200', 'Normal sinus rhythm'),
        ('201', 'Sinus rhythm with frequent PVCs'),
        ('203', 'Ventricular flutter'),
        ('212', 'Right bundle branch block'),
        ('231', 'Left bundle branch block')
    ]
    
    for record, description in featured_records:
        print(f"   • {record}: {description}")
    
    print(f"\n   All available records: {', '.join(mitdb_records)}")
    
    # Atrial fibrillation records
    afdb_records = real_loader.get_available_records('afdb')
    print(f"\n🫀 Atrial Fibrillation Database ({len(afdb_records)} records):")
    print(f"   Records: {', '.join(afdb_records[:10])}...")
    
    print(f"\n💡 Usage examples:")
    print(f"   python main.py --mode real --record 100  # Normal rhythm")
    print(f"   python main.py --mode real --record 106  # PVCs")
    print(f"   python main.py --mode real --record 101  # Atrial fibrillation")

def main():
    """Main execution function"""
    
    # Parse command line arguments
    parser = argparse.ArgumentParser(description='ECG Signal Processing System v2.0')
    parser.add_argument('--mode', choices=['demo', 'file', 'mitbih', 'real', 'realtime', 'test', 'list'], 
                       default='demo', help='Processing mode')
    parser.add_argument('--input', type=str, help='Input file path (for file mode)')
    parser.add_argument('--record', type=str, help='Record name (for mitbih/real mode)')
    parser.add_argument('--duration', type=float, default=60, 
                       help='Duration for demo mode (seconds)')
    parser.add_argument('--noise', type=float, default=0.03, 
                       help='Noise level for demo mode (0-1)')
    parser.add_argument('--heart-rate', type=int, default=75,
                       help='Target heart rate for demo mode (bpm)')
    parser.add_argument('--output', type=str, default='results', 
                       help='Output directory')
    parser.add_argument('--database', type=str, default='mitdb',
                       help='Database name for real ECG data (mitdb, afdb, etc.)')
    parser.add_argument('--test-abnormalities', action='store_true', 
                       help='Test abnormality detection with various conditions')
    
    args = parser.parse_args()
    
    # Welcome message
    print("🫀 ECG Signal Processing System v2.0")
    print("=" * 50)
    print("✨ Features: Synthetic ECG, Real Medical Data, Abnormality Detection")
    
    # Setup directories
    print("\n📁 Setting up directories...")
    setup_directories()
    
    # Ensure output directory exists
    os.makedirs(args.output, exist_ok=True)
    
    try:
        # Check for special modes first
        if args.test_abnormalities:
            test_abnormality_detection()
            return
        
        # Execute based on mode
        if args.mode == 'demo':
            print(f"🎮 Running demonstration mode...")
            results = analyze_synthetic_ecg(
                duration=args.duration, 
                noise_level=args.noise, 
                output_dir=args.output,
                heart_rate=args.heart_rate
            )
            
        elif args.mode == 'file':
            if not args.input:
                print("❌ Error: --input required for file mode")
                return
            if not os.path.exists(args.input):
                print(f"❌ Error: File {args.input} not found")
                return
            
            results = analyze_ecg_file(args.input, args.output)
            
        elif args.mode == 'mitbih':
            if not args.record:
                print("❌ Error: --record required for mitbih mode")
                return
            
            results = analyze_mitbih_record(args.record, args.output)
            
        elif args.mode == 'real':
            if not args.record:
                print("❌ Error: --record required for real mode")
                print("💡 Example: python main.py --mode real --record 100")
                print("💡 Use --mode list to see available records")
                return
            
            results = analyze_real_ecg_data(args.record, args.database, args.output)
            
        elif args.mode == 'realtime':
            run_realtime_demo()
            return
            
        elif args.mode == 'test':
            test_abnormality_detection()
            return
            
        elif args.mode == 'list':
            list_available_records()
            return
        
        # Check if analysis was successful
        if results is None:
            print("❌ Analysis failed")
            return
        
        print("\n🎉 Analysis completed successfully!")
        print(f"📁 Results saved in: {args.output}/")
        print(f"📊 Analysis plots generated")
        print(f"📋 Clinical report created")
        
        # Show file locations
        if 'files' in results:
            print("\n📄 Generated files:")
            for file_type, file_path in results['files'].items():
                print(f"   • {file_type}: {file_path}")
        
        # Show expert comparison if available
        if 'expert_summary' in results:
            expert = results['expert_summary']
            print(f"\n👨‍⚕️ Expert Comparison:")
            print(f"   • Record: {results['record_info']['description']}")
            print(f"   • Expert labeled {expert['abnormal_percentage']:.1f}% of beats as abnormal")
            print(f"   • Total annotations: {expert['total_beats']} beats")
            
            if expert['abnormal_beats'] > 0:
                print(f"   • Most common abnormalities in expert labels:")
                sorted_annotations = sorted(expert['annotation_types'].items(), 
                                          key=lambda x: x[1], reverse=True)
                for ann_type, count in sorted_annotations[:5]:
                    if count > 0 and ann_type != 'Normal':
                        print(f"     - {ann_type}: {count} beats")
        
        print("\n💡 Usage Tips:")
        print("   • Open the PNG files to view analysis plots")
        print("   • Read the TXT report for detailed findings")
        print("   • Use --mode real --record 106 for PVC examples")
        print("   • Use --mode real --record 101 for atrial fibrillation")
        print("   • Use --test-abnormalities to test detection")
        print("   • Use --mode list to see all available records")
        print("   • Use --help for more options")
        
    except Exception as e:
        print(f"❌ Error during execution: {e}")
        import traceback
        traceback.print_exc()
        return
    
    print("\n✅ ECG processing system execution completed!")

if __name__ == "__main__":
    main()