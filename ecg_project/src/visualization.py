# src/visualization.py
import numpy as np
import matplotlib.pyplot as plt
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import plotly.express as px

class ECGVisualizer:
    def __init__(self, sampling_rate=360):
        self.fs = sampling_rate
    
    def plot_ecg_analysis(self, ecg_signal, analysis_results, time_vector=None):
        """
        Create comprehensive ECG analysis plot
        """
        if time_vector is None:
            time_vector = np.arange(len(ecg_signal)) / self.fs
        
        # Ensure time_vector matches ECG signal length
        if len(time_vector) != len(ecg_signal):
            time_vector = np.arange(len(ecg_signal)) / self.fs
        
        r_peaks = analysis_results['r_peaks']
        detection_signal = analysis_results['detection_signal']
        heart_rate = analysis_results.get('heart_rate_instant', [])
        
        # Create detection signal time vector (may be different length)
        detection_time = np.arange(len(detection_signal)) / self.fs
        
        fig, axes = plt.subplots(4, 1, figsize=(15, 12))
        
        # Plot 1: ECG signal with R-peaks
        axes[0].plot(time_vector, ecg_signal, 'b-', linewidth=1, label='ECG Signal')
        
        # Plot R-peaks (ensure indices are within bounds)
        valid_peaks = r_peaks[r_peaks < len(ecg_signal)]
        if len(valid_peaks) > 0:
            axes[0].plot(time_vector[valid_peaks], ecg_signal[valid_peaks], 'ro', 
                        markersize=8, label='R-peaks')
        
        axes[0].set_title('ECG Signal with Detected R-peaks')
        axes[0].set_ylabel('Amplitude')
        axes[0].legend()
        axes[0].grid(True, alpha=0.3)
        
        # Plot 2: Detection signal (use its own time vector)
        axes[1].plot(detection_time, detection_signal, 'g-', linewidth=1)
        axes[1].set_title('Pan-Tompkins Detection Signal')
        axes[1].set_ylabel('Amplitude')
        axes[1].grid(True, alpha=0.3)
        
        # Plot 3: Heart rate
        if len(heart_rate) > 0 and len(valid_peaks) > 1:
            # Heart rate time corresponds to R-peak times (except first)
            hr_peaks = valid_peaks[1:len(heart_rate)+1]  # Match heart rate length
            if len(hr_peaks) > 0:
                hr_time = time_vector[hr_peaks]
                axes[2].plot(hr_time, heart_rate[:len(hr_peaks)], 'r-', linewidth=2, marker='o')
                axes[2].axhline(y=60, color='k', linestyle='--', alpha=0.5, label='Bradycardia')
                axes[2].axhline(y=100, color='k', linestyle='--', alpha=0.5, label='Tachycardia')
                axes[2].set_title('Instantaneous Heart Rate')
                axes[2].set_ylabel('Heart Rate (bpm)')
                axes[2].legend()
                axes[2].grid(True, alpha=0.3)
        else:
            axes[2].text(0.5, 0.5, 'Insufficient data for heart rate plot', 
                        transform=axes[2].transAxes, ha='center', va='center')
            axes[2].set_title('Instantaneous Heart Rate')
        
        # Plot 4: RR intervals
        if len(analysis_results['rr_intervals']) > 0 and len(valid_peaks) > 1:
            rr_intervals = analysis_results['rr_intervals']
            # RR interval times correspond to second R-peak onwards
            rr_peaks = valid_peaks[1:len(rr_intervals)+1]
            if len(rr_peaks) > 0:
                rr_time = time_vector[rr_peaks]
                axes[3].plot(rr_time, rr_intervals[:len(rr_peaks)] * 1000, 
                            'purple', linewidth=2, marker='s')
                axes[3].set_title('RR Intervals')
                axes[3].set_ylabel('RR Interval (ms)')
                axes[3].set_xlabel('Time (s)')
                axes[3].grid(True, alpha=0.3)
        else:
            axes[3].text(0.5, 0.5, 'Insufficient data for RR interval plot', 
                        transform=axes[3].transAxes, ha='center', va='center')
            axes[3].set_title('RR Intervals')
            axes[3].set_xlabel('Time (s)')
        
        plt.tight_layout()
        return fig
    
    def plot_qrs_morphology(self, qrs_template, individual_qrs=None):
        """
        Plot QRS morphology analysis
        """
        fig, axes = plt.subplots(1, 2, figsize=(12, 5))
        
        # Plot QRS template
        time_qrs = np.arange(len(qrs_template)) / self.fs * 1000  # Convert to ms
        axes[0].plot(time_qrs, qrs_template, 'b-', linewidth=3, label='Average QRS')
        axes[0].set_title('QRS Template')
        axes[0].set_xlabel('Time (ms)')
        axes[0].set_ylabel('Amplitude')
        axes[0].grid(True, alpha=0.3)
        axes[0].legend()
        
        # Plot individual QRS complexes (if provided)
        if individual_qrs is not None:
            for i, qrs in enumerate(individual_qrs[:10]):  # Show first 10
                if len(qrs) == len(qrs_template):  # Ensure same length
                    axes[1].plot(time_qrs, qrs, alpha=0.3, linewidth=1)
            axes[1].plot(time_qrs, qrs_template, 'r-', linewidth=3, label='Average')
            axes[1].set_title('Individual QRS Complexes')
            axes[1].set_xlabel('Time (ms)')
            axes[1].set_ylabel('Amplitude')
            axes[1].grid(True, alpha=0.3)
            axes[1].legend()
        else:
            axes[1].text(0.5, 0.5, 'No individual QRS data available', 
                        transform=axes[1].transAxes, ha='center', va='center')
            axes[1].set_title('Individual QRS Complexes')
            axes[1].set_xlabel('Time (ms)')
            axes[1].set_ylabel('Amplitude')
            axes[1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        return fig
    
    def plot_hrv_analysis(self, rr_intervals, hrv_features):
        """
        Plot Heart Rate Variability analysis
        """
        fig, axes = plt.subplots(2, 2, figsize=(12, 10))
        
        if len(rr_intervals) == 0:
            # Handle case with no RR intervals
            for i in range(2):
                for j in range(2):
                    axes[i, j].text(0.5, 0.5, 'Insufficient RR interval data', 
                                   transform=axes[i, j].transAxes, ha='center', va='center')
            
            axes[0, 0].set_title('RR Interval Time Series')
            axes[0, 1].set_title('RR Interval Distribution')
            axes[1, 0].set_title('Poincaré Plot')
            axes[1, 1].set_title('HRV Features')
            
            plt.tight_layout()
            return fig
        
        # RR interval time series
        axes[0, 0].plot(rr_intervals * 1000, 'b-', marker='o', markersize=4)
        axes[0, 0].set_title('RR Interval Time Series')
        axes[0, 0].set_xlabel('Beat Number')
        axes[0, 0].set_ylabel('RR Interval (ms)')
        axes[0, 0].grid(True, alpha=0.3)
        
        # RR interval histogram
        if len(rr_intervals) > 1:
            axes[0, 1].hist(rr_intervals * 1000, bins=min(20, len(rr_intervals)//2), 
                           alpha=0.7, edgecolor='black')
        axes[0, 1].set_title('RR Interval Distribution')
        axes[0, 1].set_xlabel('RR Interval (ms)')
        axes[0, 1].set_ylabel('Frequency')
        axes[0, 1].grid(True, alpha=0.3)
        
        # Poincaré plot
        if len(rr_intervals) > 1:
            rr1 = rr_intervals[:-1] * 1000
            rr2 = rr_intervals[1:] * 1000
            axes[1, 0].scatter(rr1, rr2, alpha=0.6, s=20)
            # Add identity line
            min_rr, max_rr = min(np.min(rr1), np.min(rr2)), max(np.max(rr1), np.max(rr2))
            axes[1, 0].plot([min_rr, max_rr], [min_rr, max_rr], 'r--', alpha=0.5)
            axes[1, 0].set_xlabel('RR(n) (ms)')
            axes[1, 0].set_ylabel('RR(n+1) (ms)')
        else:
            axes[1, 0].text(0.5, 0.5, 'Need >1 RR interval for Poincaré plot', 
                           transform=axes[1, 0].transAxes, ha='center', va='center')
        
        axes[1, 0].set_title('Poincaré Plot')
        axes[1, 0].grid(True, alpha=0.3)
        
        # HRV features bar plot
        hrv_metrics = ['rmssd', 'pnn50', 'triangular_index']
        hrv_values = [hrv_features.get(metric, 0) for metric in hrv_metrics]
        hrv_labels = ['RMSSD\n(ms)', 'pNN50\n(%)', 'Triangular\nIndex']
        
        bars = axes[1, 1].bar(hrv_labels, hrv_values, alpha=0.7, 
                             color=['skyblue', 'lightgreen', 'salmon'])
        axes[1, 1].set_title('HRV Features')
        axes[1, 1].set_ylabel('Value')
        axes[1, 1].grid(True, alpha=0.3, axis='y')
        
        # Add value labels on bars
        for bar, value in zip(bars, hrv_values):
            height = bar.get_height()
            axes[1, 1].text(bar.get_x() + bar.get_width()/2., height + height*0.01,
                           f'{value:.1f}', ha='center', va='bottom')
        
        plt.tight_layout()
        return fig
    
    def create_interactive_dashboard(self, ecg_signal, analysis_results, time_vector=None):
        """
        Create interactive dashboard using Plotly
        """
        if time_vector is None:
            time_vector = np.arange(len(ecg_signal)) / self.fs
        
        # Ensure time_vector matches ECG signal length
        if len(time_vector) != len(ecg_signal):
            time_vector = np.arange(len(ecg_signal)) / self.fs
        
        # Create subplots
        fig = make_subplots(
            rows=3, cols=1,
            subplot_titles=('ECG Signal with R-peaks', 'Heart Rate', 'RR Intervals'),
            vertical_spacing=0.08
        )
        
        # ECG signal
        fig.add_trace(
            go.Scatter(x=time_vector, y=ecg_signal, mode='lines', 
                      name='ECG Signal', line=dict(color='blue', width=1)),
            row=1, col=1
        )
        
        # R-peaks
        r_peaks = analysis_results['r_peaks']
        valid_peaks = r_peaks[r_peaks < len(ecg_signal)]
        
        if len(valid_peaks) > 0:
            fig.add_trace(
                go.Scatter(x=time_vector[valid_peaks], y=ecg_signal[valid_peaks], 
                          mode='markers', name='R-peaks', 
                          marker=dict(color='red', size=8)),
                row=1, col=1
            )
        
        # Heart rate
        heart_rate = analysis_results.get('heart_rate_instant', [])
        if len(heart_rate) > 0 and len(valid_peaks) > 1:
            hr_peaks = valid_peaks[1:len(heart_rate)+1]
            if len(hr_peaks) > 0:
                hr_time = time_vector[hr_peaks]
                fig.add_trace(
                    go.Scatter(x=hr_time, y=heart_rate[:len(hr_peaks)], 
                              mode='lines+markers', name='Heart Rate',
                              line=dict(color='red', width=2)),
                    row=2, col=1
                )
        
        # RR intervals
        rr_intervals = analysis_results['rr_intervals']
        if len(rr_intervals) > 0 and len(valid_peaks) > 1:
            rr_peaks = valid_peaks[1:len(rr_intervals)+1]
            if len(rr_peaks) > 0:
                rr_time = time_vector[rr_peaks]
                fig.add_trace(
                    go.Scatter(x=rr_time, y=rr_intervals[:len(rr_peaks)] * 1000, 
                              mode='lines+markers', name='RR Intervals',
                              line=dict(color='purple', width=2)),
                    row=3, col=1
                )
        
        # Update layout
        fig.update_layout(
            title='ECG Analysis Dashboard',
            height=800,
            showlegend=True
        )
        
        fig.update_xaxes(title_text="Time (s)", row=3, col=1)
        fig.update_yaxes(title_text="Amplitude", row=1, col=1)
        fig.update_yaxes(title_text="Heart Rate (bpm)", row=2, col=1)
        fig.update_yaxes(title_text="RR Interval (ms)", row=3, col=1)
        
        return fig
    
    def plot_comparison(self, signals_dict, titles=None):
        """
        Plot multiple ECG signals for comparison
        """
        n_signals = len(signals_dict)
        fig, axes = plt.subplots(n_signals, 1, figsize=(15, 3*n_signals))
        
        if n_signals == 1:
            axes = [axes]
        
        for i, (name, signal) in enumerate(signals_dict.items()):
            time_vec = np.arange(len(signal)) / self.fs
            axes[i].plot(time_vec, signal, linewidth=1)
            axes[i].set_title(titles[i] if titles else name)
            axes[i].set_ylabel('Amplitude')
            axes[i].grid(True, alpha=0.3)
            
            if i == n_signals - 1:
                axes[i].set_xlabel('Time (s)')
        
        plt.tight_layout()
        return fig


# Test visualization
if __name__ == "__main__":
    import sys
    import os
    sys.path.append(os.path.dirname(__file__))
    
    try:
        from preprocessing import ECGDataLoader, ECGPreprocessor
        from detection import QRSDetector
        from features import ECGFeatureExtractor
        
        print("Testing ECG visualization...")
        
        # Initialize classes
        loader = ECGDataLoader()
        preprocessor = ECGPreprocessor()
        detector = QRSDetector()
        extractor = ECGFeatureExtractor()
        visualizer = ECGVisualizer()
        
        # Generate and process data
        print("1. Generating test data...")
        data = loader.generate_realistic_ecg(duration=20, noise_level=0.02)
        clean_signal = preprocessor.preprocess_signal(data['signal'])
        
        print("2. Analyzing ECG...")
        qrs_results = detector.detect_qrs_with_metrics(clean_signal)
        features = extractor.extract_all_features(
            clean_signal, qrs_results['r_peaks'], qrs_results['rr_intervals']
        )
        
        # Create visualizations
        analysis_results = {**qrs_results, 'features': features}
        
        print("3. Creating visualizations...")
        
        # Main analysis plot
        fig1 = visualizer.plot_ecg_analysis(clean_signal, analysis_results, data['time'])
        plt.show()
        
        # HRV analysis plot
        fig2 = visualizer.plot_hrv_analysis(qrs_results['rr_intervals'], features)
        plt.show()
        
        print("✅ Visualization test completed successfully!")
        
    except ImportError as e:
        print(f"Import error: {e}")
    except Exception as e:
        print(f"Error during testing: {e}")
        import traceback
        traceback.print_exc()