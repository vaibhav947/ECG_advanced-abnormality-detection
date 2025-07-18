# src/realtime.py
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import time
from collections import deque
import threading
import queue

class RealTimeECGProcessor:
    def __init__(self, sampling_rate=360, buffer_size=1800):  # 5 seconds buffer
        self.fs = sampling_rate
        self.buffer_size = buffer_size
        self.buffer = deque(maxlen=buffer_size)
        self.processed_buffer = deque(maxlen=buffer_size)
        self.r_peaks = deque(maxlen=50)  # Store last 50 R-peaks
        self.heart_rates = deque(maxlen=50)
        
        # Processing components
        from preprocessing import ECGPreprocessor
        from detection import QRSDetector
        
        self.preprocessor = ECGPreprocessor(sampling_rate)
        self.detector = QRSDetector(sampling_rate)
        
        # Real-time processing parameters
        self.processing_window = int(0.5 * sampling_rate)  # 0.5 second window
        self.last_r_peak = 0
        self.refractory_period = int(0.2 * sampling_rate)  # 200ms refractory period
        
        # Alerts and monitoring
        self.alerts = []
        self.monitoring = False
        
    def add_sample(self, sample):
        """Add a new ECG sample to the buffer"""
        self.buffer.append(sample)
        
        # Process if we have enough samples
        if len(self.buffer) >= self.processing_window:
            self.process_latest_window()
    
    def add_samples(self, samples):
        """Add multiple ECG samples to the buffer"""
        for sample in samples:
            self.add_sample(sample)
    
    def process_latest_window(self):
        """Process the latest window of ECG data"""
        if len(self.buffer) < self.processing_window:
            return
        
        # Get latest window
        window_data = np.array(list(self.buffer)[-self.processing_window:])
        
        # Preprocess
        processed_window = self.preprocessor.preprocess_signal(window_data)
        
        # Detect QRS in window
        peaks, _ = self.detector.pan_tompkins_qrs_detect(processed_window)
        
        # Adjust peak positions to absolute time
        current_time = len(self.buffer)
        absolute_peaks = peaks + (current_time - self.processing_window)
        
        # Filter peaks based on refractory period
        valid_peaks = []
        for peak in absolute_peaks:
            if peak > self.last_r_peak + self.refractory_period:
                valid_peaks.append(peak)
                self.last_r_peak = peak
        
        # Add valid peaks to buffer
        for peak in valid_peaks:
            self.r_peaks.append(peak)
            
            # Calculate heart rate
            if len(self.r_peaks) > 1:
                rr_interval = (self.r_peaks[-1] - self.r_peaks[-2]) / self.fs
                heart_rate = 60 / rr_interval
                self.heart_rates.append(heart_rate)
                
                # Check for abnormalities
                self.check_abnormalities(heart_rate)
        
        # Store processed data
        self.processed_buffer.extend(processed_window)
    
    def check_abnormalities(self, heart_rate):
        """Check for real-time abnormalities"""
        current_time = time.time()
        
        # Bradycardia
        if heart_rate < 60:
            self.alerts.append({
                'type': 'Bradycardia',
                'value': heart_rate,
                'time': current_time,
                'severity': 'Medium' if heart_rate < 50 else 'Low'
            })
        
        # Tachycardia
        if heart_rate > 100:
            self.alerts.append({
                'type': 'Tachycardia',
                'value': heart_rate,
                'time': current_time,
                'severity': 'High' if heart_rate > 150 else 'Medium'
            })
        
        # Irregular rhythm (simplified)
        if len(self.heart_rates) > 5:
            recent_hr = list(self.heart_rates)[-5:]
            hr_std = np.std(recent_hr)
            if hr_std > 20:  # High variability
                self.alerts.append({
                    'type': 'Irregular Rhythm',
                    'value': hr_std,
                    'time': current_time,
                    'severity': 'Medium'
                })
    
    def get_current_stats(self):
        """Get current ECG statistics"""
        stats = {
            'heart_rate': list(self.heart_rates)[-1] if self.heart_rates else 0,
            'average_hr': np.mean(self.heart_rates) if self.heart_rates else 0,
            'hr_variability': np.std(self.heart_rates) if len(self.heart_rates) > 1 else 0,
            'total_beats': len(self.r_peaks),
            'recent_alerts': [alert for alert in self.alerts if time.time() - alert['time'] < 30]
        }
        return stats
    
    def simulate_realtime_data(self, duration=30):
        """Simulate real-time ECG data stream"""
        from preprocessing import ECGDataLoader
        
        # Generate synthetic ECG data
        loader = ECGDataLoader(self.fs)
        data = loader.generate_sample_ecg(duration=duration, noise_level=0.03)
        
        # Simulate real-time streaming
        samples_per_chunk = int(0.1 * self.fs)  # 100ms chunks
        
        for i in range(0, len(data['signal']), samples_per_chunk):
            chunk = data['signal'][i:i+samples_per_chunk]
            self.add_samples(chunk)
            
            # Print current stats every 5 seconds
            if i % (5 * self.fs) == 0:
                stats = self.get_current_stats()
                print(f"Time: {i/self.fs:.1f}s, HR: {stats['heart_rate']:.1f} bpm, "
                      f"Avg HR: {stats['average_hr']:.1f} bpm, Beats: {stats['total_beats']}")
                
                # Print alerts
                for alert in stats['recent_alerts']:
                    if time.time() - alert['time'] < 1:  # New alert
                        print(f"ALERT: {alert['type']} - {alert['value']:.1f} ({alert['severity']})")
            
            # Simulate real-time delay
            time.sleep(0.01)  # 10ms delay
    
    def create_realtime_plot(self):
        """Create real-time plotting window"""
        fig, axes = plt.subplots(3, 1, figsize=(12, 10))
        
        # Initialize empty plots
        line_ecg, = axes[0].plot([], [], 'b-', linewidth=1)
        line_peaks, = axes[0].plot([], [], 'ro', markersize=6)
        line_hr, = axes[1].plot([], [], 'r-', linewidth=2)
        line_alerts, = axes[2].plot([], [], 'orange', linewidth=2)
        
        # Set up axes
        axes[0].set_title('Real-time ECG Signal')
        axes[0].set_ylabel('Amplitude')
        axes[0].grid(True, alpha=0.3)
        
        axes[1].set_title('Heart Rate')
        axes[1].set_ylabel('HR (bpm)')
        axes[1].axhline(y=60, color='k', linestyle='--', alpha=0.5)
        axes[1].axhline(y=100, color='k', linestyle='--', alpha=0.5)
        axes[1].grid(True, alpha=0.3)
        
        axes[2].set_title('Alerts Timeline')
        axes[2].set_ylabel('Alert Count')
        axes[2].set_xlabel('Time (s)')
        axes[2].grid(True, alpha=0.3)
        
        def update_plot(frame):
            if len(self.buffer) > 0:
                # ECG signal
                time_vec = np.arange(len(self.buffer)) / self.fs
                line_ecg.set_data(time_vec, list(self.buffer))
                
                # R-peaks
                if self.r_peaks:
                    peak_times = np.array(list(self.r_peaks)) / self.fs
                    peak_values = [list(self.buffer)[int(peak)] for peak in self.r_peaks 
                                 if int(peak) < len(self.buffer)]
                    line_peaks.set_data(peak_times[-len(peak_values):], peak_values)
                
                # Heart rate
                if self.heart_rates:
                    hr_times = np.arange(len(self.heart_rates)) * 0.5  # Approximate timing
                    line_hr.set_data(hr_times, list(self.heart_rates))
                
                # Update axis limits
                if len(self.buffer) > 0:
                    axes[0].set_xlim(max(0, len(self.buffer)/self.fs - 10), len(self.buffer)/self.fs)
                    axes[0].set_ylim(min(self.buffer) - 0.1, max(self.buffer) + 0.1)
                
                if self.heart_rates:
                    axes[1].set_xlim(max(0, len(self.heart_rates) * 0.5 - 10), len(self.heart_rates) * 0.5)
                    axes[1].set_ylim(40, 120)
        
        # Create animation
        anim = FuncAnimation(fig, update_plot, interval=100, blit=False)
        plt.tight_layout()
        
        return fig, anim

# Test real-time processing
if __name__ == "__main__":
    # Initialize real-time processor
    processor = RealTimeECGProcessor()
    
    print("Starting real-time ECG simulation...")
    print("Monitoring for 30 seconds...")
    
    # Start real-time processing in a separate thread
    processing_thread = threading.Thread(
        target=processor.simulate_realtime_data,
        args=(30,)
    )
    processing_thread.start()
    
    # Wait for processing to complete
    processing_thread.join()
    
    print("\nFinal Statistics:")
    final_stats = processor.get_current_stats()
    print(f"Total beats detected: {final_stats['total_beats']}")
    print(f"Average heart rate: {final_stats['average_hr']:.1f} bpm")
    print(f"Heart rate variability: {final_stats['hr_variability']:.1f} bpm")
    print(f"Total alerts: {len(processor.alerts)}")
    
    # Show alerts
    if processor.alerts:
        print("\nAlerts generated:")
        for alert in processor.alerts:
            print(f"  {alert['type']}: {alert['value']:.1f} ({alert['severity']})")