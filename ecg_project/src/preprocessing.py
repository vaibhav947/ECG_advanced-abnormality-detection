# src/preprocessing.py
import numpy as np
import pandas as pd
import os
from scipy import signal
import matplotlib.pyplot as plt

# Handle different SciPy versions for gaussian function
try:
    from scipy.signal.windows import gaussian
except ImportError:
    try:
        from scipy.signal import gaussian
    except ImportError:
        # Fallback implementation if gaussian is not available
        def gaussian(M, std):
            n = np.arange(0, M) - (M - 1.0) / 2.0
            sig2 = 2 * std * std
            w = np.exp(-(n ** 2) / sig2)
            return w

class ECGDataLoader:
    def __init__(self, sampling_rate=360):
        self.sampling_rate = sampling_rate
        
    def load_mitbih_record(self, record_name, database_path='mitdb'):
        """Load ECG record from MIT-BIH database (old method - kept for compatibility)"""
        try:
            import wfdb
            # Download from PhysioNet if not available locally
            # Try both parameter names for compatibility
            try:
                record = wfdb.rdrecord(record_name, pn_dir=database_path)
                annotation = wfdb.rdann(record_name, 'atr', pn_dir=database_path)
            except TypeError:
                # Fallback to older parameter name
                record = wfdb.rdrecord(record_name, pb_dir=database_path)
                annotation = wfdb.rdann(record_name, 'atr', pb_dir=database_path)
            
            # Extract signal (usually channel 0 is MLII)
            ecg_signal = record.p_signal[:, 0]
            
            return {
                'signal': ecg_signal,
                'sampling_rate': record.fs,
                'annotations': annotation,
                'record_name': record_name
            }
        except Exception as e:
            print(f"Error loading record {record_name}: {e}")
            return None
    
    def generate_sample_ecg(self, duration=10, noise_level=0.05, heart_rate=75):
        """Generate a synthetic ECG signal with specified heart rate"""
        t = np.linspace(0, duration, int(duration * self.sampling_rate))
        
        # Simple ECG model with custom heart rate
        ecg = np.zeros_like(t)
        period = 60 / heart_rate  # seconds - now uses parameter
        
        # Add some variability for more realistic rhythm
        variability = 0.1 if heart_rate > 60 else 0.05  # Less variability for bradycardia
        
        current_time = 0
        while current_time < duration:
            # Add slight heart rate variability
            current_period = period + np.random.normal(0, period * variability)
            current_period = max(0.3, min(2.0, current_period))  # Bounds checking
            
            if current_time + 0.1 < duration:
                # QRS complex simulation
                qrs_start = current_time
                qrs_duration = 0.1
                qrs_indices = np.where((t >= qrs_start) & (t <= qrs_start + qrs_duration))[0]
                
                if len(qrs_indices) > 0:
                    try:
                        qrs_shape = gaussian(len(qrs_indices), std=5)
                        ecg[qrs_indices] = qrs_shape
                    except:
                        # Fallback if gaussian fails
                        qrs_shape = np.exp(-(np.arange(len(qrs_indices)) - len(qrs_indices)//2)**2 / (2*5**2))
                        ecg[qrs_indices] = qrs_shape
            
            current_time += current_period
        
        # Add noise
        noise = np.random.normal(0, noise_level, len(ecg))
        ecg_noisy = ecg + noise
        
        return {
            'signal': ecg_noisy,
            'sampling_rate': self.sampling_rate,
            'time': t,
            'clean_signal': ecg,
            'target_heart_rate': heart_rate
        }
    
    def generate_realistic_ecg(self, duration=10, heart_rate=75, noise_level=0.05):
        """Generate a more realistic synthetic ECG signal"""
        fs = self.sampling_rate
        t = np.linspace(0, duration, int(duration * fs))
        ecg = np.zeros_like(t)
        
        # Heart rate parameters
        rr_interval = 60 / heart_rate  # seconds
        
        # ECG wave parameters (approximate)
        p_duration = 0.08  # P wave duration
        qrs_duration = 0.08  # QRS duration
        t_duration = 0.16  # T wave duration
        
        # Generate beats with some variability
        beat_times = []
        current_time = 0
        
        while current_time < duration:
            # Add heart rate variability
            hr_variation = np.random.normal(0, 3)  # 3 bpm standard deviation
            current_hr = max(30, min(200, heart_rate + hr_variation))
            current_rr = 60 / current_hr
            
            beat_times.append(current_time)
            current_time += current_rr
        
        # Generate ECG morphology
        for i, beat_time in enumerate(beat_times):
            if beat_time + 0.4 < duration:
                
                # P wave (small positive deflection)
                p_start = beat_time
                p_indices = np.where((t >= p_start) & (t <= p_start + p_duration))[0]
                if len(p_indices) > 0:
                    p_wave = 0.1 * np.sin(np.pi * (t[p_indices] - p_start) / p_duration)
                    ecg[p_indices] += p_wave
                
                # QRS complex (main spike)
                qrs_start = beat_time + 0.12  # PR interval
                qrs_indices = np.where((t >= qrs_start) & (t <= qrs_start + qrs_duration))[0]
                if len(qrs_indices) > 0:
                    # Create QRS shape: Q (negative), R (positive), S (negative)
                    qrs_t = (t[qrs_indices] - qrs_start) / qrs_duration
                    q_wave = -0.1 * np.exp(-((qrs_t - 0.2) / 0.1) ** 2)
                    r_wave = 1.0 * np.exp(-((qrs_t - 0.5) / 0.15) ** 2)
                    s_wave = -0.2 * np.exp(-((qrs_t - 0.8) / 0.1) ** 2)
                    ecg[qrs_indices] += q_wave + r_wave + s_wave
                
                # T wave (positive deflection)
                t_start = qrs_start + qrs_duration + 0.05  # ST segment
                t_indices = np.where((t >= t_start) & (t <= t_start + t_duration))[0]
                if len(t_indices) > 0:
                    t_wave = 0.3 * np.sin(np.pi * (t[t_indices] - t_start) / t_duration)
                    ecg[t_indices] += t_wave
        
        # Add baseline wander (low frequency noise)
        baseline_freq = 0.5  # Hz
        baseline_wander = 0.1 * np.sin(2 * np.pi * baseline_freq * t)
        
        # Add high frequency noise
        noise = np.random.normal(0, noise_level, len(ecg))
        
        # Combine all components
        ecg_final = ecg + baseline_wander + noise
        
        return {
            'signal': ecg_final,
            'sampling_rate': self.sampling_rate,
            'time': t,
            'clean_signal': ecg,
            'heart_rate': heart_rate,
            'beat_times': beat_times
        }
    
    def load_csv_data(self, filepath, signal_column='ECG', fs=360):
        """Load ECG data from CSV file"""
        try:
            df = pd.read_csv(filepath)
            ecg_signal = df[signal_column].values
            
            return {
                'signal': ecg_signal,
                'sampling_rate': fs,
                'time': np.arange(len(ecg_signal)) / fs
            }
        except Exception as e:
            print(f"Error loading CSV: {e}")
            return None


class RealECGDataLoader:
    """Loader for real ECG data from medical databases"""
    
    def __init__(self, sampling_rate=360):
        self.sampling_rate = sampling_rate
        self.data_dir = 'data/real_ecg'
        os.makedirs(self.data_dir, exist_ok=True)
    
    def load_mitbih_record(self, record_name, database='mitdb'):
        """
        Load real ECG data from MIT-BIH Arrhythmia Database
        
        Args:
            record_name (str): Record number (e.g., '100', '101', '102')
            database (str): Database name ('mitdb', 'afdb', 'edb', etc.)
        
        Returns:
            dict: ECG data with expert annotations
        """
        try:
            import wfdb
            
            print(f"📥 Downloading MIT-BIH record {record_name} from {database}...")
            
            # Download record from PhysioNet - try both parameter names for compatibility
            try:
                # Try newer parameter name first
                record = wfdb.rdrecord(record_name, pn_dir=database)
                annotation = wfdb.rdann(record_name, 'atr', pn_dir=database)
            except TypeError:
                # Fallback to older parameter name
                record = wfdb.rdrecord(record_name, pb_dir=database)
                annotation = wfdb.rdann(record_name, 'atr', pb_dir=database)
            
            # Extract ECG signal (usually lead II - channel 0)
            ecg_signal = record.p_signal[:, 0]  # Lead II (MLII)
            
            # Get annotations (expert labels)
            beat_annotations = annotation.symbol  # Beat types
            beat_locations = annotation.sample    # Sample locations
            
            # Create annotation mapping
            annotation_map = {
                'N': 'Normal',
                'L': 'Left bundle branch block',
                'R': 'Right bundle branch block', 
                'V': 'Premature ventricular contraction',
                'A': 'Atrial premature beat',
                'F': 'Fusion beat',
                'J': 'Nodal escape beat',
                'E': 'Ventricular escape beat',
                '/': 'Paced beat',
                'f': 'Fusion of paced and normal',
                'Q': 'Unclassifiable beat',
                '!': 'Ventricular flutter wave',
                '[': 'Start of ventricular flutter/fibrillation',
                ']': 'End of ventricular flutter/fibrillation',
                'x': 'Non-conducted P-wave',
                '(': 'Waveform onset',
                ')': 'Waveform end',
                'p': 'Peak of P-wave',
                't': 'Peak of T-wave',
                'u': 'Peak of U-wave',
                '`': 'PQ junction',
                "'": 'J-point',
                '^': 'Non-conducted pacer spike',
                '|': 'Isolated QRS-like artifact',
                '~': 'Change in signal quality',
                '+': 'Rhythm change',
                's': 'ST change',
                'T': 'T-wave change',
                '*': 'Systole',
                'D': 'Diastole',
                '=': 'Measurement annotation',
                '"': 'Comment annotation',
                '@': 'Link to external data'
            }
            
            # Process annotations
            processed_annotations = []
            for i, (symbol, location) in enumerate(zip(beat_annotations, beat_locations)):
                processed_annotations.append({
                    'sample': location,
                    'time': location / record.fs,
                    'symbol': symbol,
                    'description': annotation_map.get(symbol, f'Unknown ({symbol})'),
                    'is_abnormal': symbol not in ['N', '.', '+']  # Normal beats and rhythm changes
                })
            
            return {
                'signal': ecg_signal,
                'sampling_rate': record.fs,
                'time': np.arange(len(ecg_signal)) / record.fs,
                'annotations': processed_annotations,
                'record_name': record_name,
                'database': database,
                'duration': len(ecg_signal) / record.fs,
                'patient_info': {
                    'record': record_name,
                    'leads': record.sig_name,
                    'units': record.units,
                    'comments': record.comments
                }
            }
            
        except ImportError:
            print("❌ Error: wfdb package not installed")
            print("💡 Install with: pip install wfdb")
            return None
        except Exception as e:
            print(f"❌ Error loading MIT-BIH record {record_name}: {e}")
            print("💡 Make sure you have internet connection for first download")
            return None
    
    def load_csv_ecg_data(self, filepath, signal_column='ECG', annotation_column=None):
        """
        Load ECG data from CSV files (common format)
        
        Args:
            filepath (str): Path to CSV file
            signal_column (str): Column name containing ECG signal
            annotation_column (str): Column name containing annotations (optional)
        """
        try:
            print(f"📥 Loading ECG data from {filepath}...")
            
            df = pd.read_csv(filepath)
            
            # Extract ECG signal
            ecg_signal = df[signal_column].values
            
            # Extract annotations if available
            annotations = None
            if annotation_column and annotation_column in df.columns:
                annotations = df[annotation_column].values
            
            return {
                'signal': ecg_signal,
                'sampling_rate': self.sampling_rate,
                'time': np.arange(len(ecg_signal)) / self.sampling_rate,
                'annotations': annotations,
                'source': 'CSV',
                'filepath': filepath,
                'columns': list(df.columns)
            }
            
        except Exception as e:
            print(f"❌ Error loading CSV data: {e}")
            return None
    
    def get_available_records(self, database='mitdb'):
        """
        Get list of available records in a database
        
        Args:
            database (str): Database name
            
        Returns:
            list: Available record names
        """
        # Common MIT-BIH records
        mitdb_records = [
            '100', '101', '102', '103', '104', '105', '106', '107', '108', '109',
            '111', '112', '113', '114', '115', '116', '117', '118', '119', '121',
            '122', '123', '124', '200', '201', '202', '203', '205', '207', '208',
            '209', '210', '212', '213', '214', '215', '217', '219', '220', '221',
            '222', '223', '228', '230', '231', '232', '233', '234'
        ]
        
        # Atrial fibrillation database records
        afdb_records = [
            '04015', '04043', '04048', '04126', '04908', '05091', '05121', '05261',
            '06426', '06453', '06995', '07162', '07859', '07879', '07910', '08215',
            '08219', '08378', '08405', '08434', '08455'
        ]
        
        if database == 'mitdb':
            return mitdb_records
        elif database == 'afdb':
            return afdb_records
        else:
            return []
    
    def get_record_info(self, record_name, database='mitdb'):
        """
        Get information about a specific record
        
        Args:
            record_name (str): Record name
            database (str): Database name
            
        Returns:
            dict: Record information
        """
        # Known information about common records
        record_info = {
            'mitdb': {
                '100': 'Normal sinus rhythm',
                '101': 'Atrial fibrillation',
                '102': 'Sinus tachycardia with PVCs',
                '103': 'Sinus arrhythmia',
                '104': 'Sinus rhythm with PVCs and aberrant conduction',
                '105': 'Sinus rhythm with PVCs',
                '106': 'Sinus rhythm with frequent PVCs',
                '107': 'Sinus rhythm with PVCs',
                '108': 'Left bundle branch block',
                '109': 'Right bundle branch block',
                '111': 'Sinus rhythm with 2nd degree AV block',
                '112': 'Sinus rhythm with PVCs',
                '113': 'Sinus rhythm with PVCs',
                '114': 'Sinus rhythm with frequent PVCs',
                '115': 'Sinus rhythm with frequent PVCs',
                '116': 'Sinus rhythm with frequent PVCs',
                '117': 'Sinus rhythm with frequent PVCs',
                '118': 'Right bundle branch block with PVCs',
                '119': 'Sinus rhythm with frequent PVCs',
                '121': 'Sinus rhythm with frequent PVCs',
                '122': 'Sinus rhythm with frequent PVCs',
                '123': 'Sinus rhythm with frequent PVCs',
                '124': 'Sinus rhythm with frequent PVCs',
                '200': 'Normal sinus rhythm',
                '201': 'Sinus rhythm with frequent PVCs',
                '202': 'Sinus rhythm with frequent PVCs',
                '203': 'Ventricular flutter',
                '205': 'Sinus rhythm with frequent PVCs',
                '207': 'Sinus rhythm with frequent PVCs',
                '208': 'Sinus rhythm with frequent PVCs',
                '209': 'Sinus rhythm with frequent PVCs',
                '210': 'Sinus rhythm with frequent PVCs',
                '212': 'Right bundle branch block',
                '213': 'Sinus rhythm with frequent PVCs',
                '214': 'Sinus rhythm with frequent PVCs',
                '215': 'Sinus rhythm with frequent PVCs',
                '217': 'Sinus rhythm with frequent PVCs',
                '219': 'Right bundle branch block with PVCs',
                '220': 'Normal sinus rhythm',
                '221': 'Sinus rhythm with frequent PVCs',
                '222': 'Sinus rhythm with frequent PVCs',
                '223': 'Sinus rhythm with frequent PVCs',
                '228': 'Sinus rhythm with frequent PVCs',
                '230': 'Sinus rhythm with frequent PVCs',
                '231': 'Left bundle branch block',
                '232': 'Normal sinus rhythm',
                '233': 'Sinus rhythm with frequent PVCs',
                '234': 'Sinus rhythm with frequent PVCs'
            }
        }
        
        return {
            'record': record_name,
            'database': database,
            'description': record_info.get(database, {}).get(record_name, 'Unknown condition'),
            'available': record_name in self.get_available_records(database)
        }


class ECGPreprocessor:
    def __init__(self, sampling_rate=360):
        self.fs = sampling_rate
        self.nyquist = sampling_rate / 2
    
    def remove_baseline_wander(self, ecg_signal, cutoff=0.5):
        """Remove baseline wander using high-pass filter"""
        # Design high-pass filter
        b, a = signal.butter(3, cutoff / self.nyquist, btype='high')
        filtered_signal = signal.filtfilt(b, a, ecg_signal)
        return filtered_signal
    
    def bandpass_filter(self, ecg_signal, low_freq=0.5, high_freq=40):
        """Apply bandpass filter to ECG signal"""
        # Design bandpass filter
        b, a = signal.butter(3, [low_freq / self.nyquist, high_freq / self.nyquist], 
                           btype='band')
        filtered_signal = signal.filtfilt(b, a, ecg_signal)
        return filtered_signal
    
    def notch_filter(self, ecg_signal, notch_freq=50, quality=30):
        """Remove power line interference (50Hz or 60Hz)"""
        # Design notch filter
        b, a = signal.iirnotch(notch_freq / self.nyquist, quality)
        filtered_signal = signal.filtfilt(b, a, ecg_signal)
        return filtered_signal
    
    def preprocess_signal(self, ecg_signal, apply_notch=True, notch_freq=50):
        """Complete preprocessing pipeline"""
        # Step 1: Remove baseline wander
        step1 = self.remove_baseline_wander(ecg_signal)
        
        # Step 2: Apply bandpass filter
        step2 = self.bandpass_filter(step1)
        
        # Step 3: Apply notch filter (optional)
        if apply_notch:
            step3 = self.notch_filter(step2, notch_freq)
        else:
            step3 = step2
        
        return step3
    
    def signal_quality_assessment(self, ecg_signal):
        """Assess signal quality"""
        # Calculate signal-to-noise ratio
        signal_power = np.mean(ecg_signal**2)
        noise_estimate = np.std(np.diff(ecg_signal))
        snr = 10 * np.log10(signal_power / noise_estimate**2) if noise_estimate > 0 else 0
        
        # Check for clipping (fixed logic)
        signal_range = np.max(ecg_signal) - np.min(ecg_signal)
        max_expected = 3.0  # Expected maximum range for normalized ECG
        is_clipped = signal_range > max_expected
        
        # Check for flat segments
        flat_segments = np.sum(np.diff(ecg_signal) == 0) / len(ecg_signal) if len(ecg_signal) > 1 else 0
        
        quality_score = {
            'snr': float(snr),  # Convert to regular Python float
            'is_clipped': bool(is_clipped),  # Convert to regular Python bool
            'flat_percentage': float(flat_segments * 100),  # Convert to regular Python float
            'quality': 'Good' if snr > 10 and not is_clipped and flat_segments < 0.1 else 'Poor'
        }
        
        return quality_score


# Enhanced ECG Generator for abnormality testing
class EnhancedECGGenerator:
    def __init__(self, sampling_rate=360):
        self.fs = sampling_rate
    
    def generate_abnormal_ecg(self, duration=60, abnormality_type='bradycardia', severity='mild'):
        """
        Generate ECG with specific abnormalities for testing
        
        Args:
            duration: signal duration in seconds
            abnormality_type: 'bradycardia', 'tachycardia', 'arrhythmia', 'atrial_fib', 'pvcs'
            severity: 'mild', 'moderate', 'severe'
        """
        t = np.linspace(0, duration, int(duration * self.fs))
        ecg = np.zeros_like(t)
        
        # Base parameters
        if abnormality_type == 'bradycardia':
            if severity == 'mild':
                base_hr = 55  # Mild bradycardia
            elif severity == 'moderate':
                base_hr = 45  # Moderate bradycardia
            else:
                base_hr = 35  # Severe bradycardia
            hr_variability = 2
            
        elif abnormality_type == 'tachycardia':
            if severity == 'mild':
                base_hr = 110  # Mild tachycardia
            elif severity == 'moderate':
                base_hr = 130  # Moderate tachycardia
            else:
                base_hr = 160  # Severe tachycardia
            hr_variability = 5
            
        elif abnormality_type == 'arrhythmia':
            base_hr = 75
            if severity == 'mild':
                hr_variability = 15  # Mild irregularity
            elif severity == 'moderate':
                hr_variability = 25  # Moderate irregularity
            else:
                hr_variability = 40  # Severe irregularity
                
        elif abnormality_type == 'atrial_fib':
            base_hr = 85
            hr_variability = 30  # Very irregular
            
        elif abnormality_type == 'pvcs':
            base_hr = 70
            hr_variability = 5
            
        else:  # normal
            base_hr = 72
            hr_variability = 3
        
        # Generate beats with variability
        current_time = 0
        beat_times = []
        
        while current_time < duration:
            # Add heart rate variability
            if abnormality_type == 'atrial_fib':
                # Atrial fibrillation: very irregular
                hr_variation = np.random.normal(0, hr_variability)
                current_hr = max(30, min(200, base_hr + hr_variation))
            elif abnormality_type == 'arrhythmia':
                # Irregular rhythm
                hr_variation = np.random.normal(0, hr_variability)
                current_hr = max(40, min(150, base_hr + hr_variation))
            else:
                # Normal variability
                hr_variation = np.random.normal(0, hr_variability)
                current_hr = max(30, min(200, base_hr + hr_variation))
            
            rr_interval = 60 / current_hr
            
            # Add PVCs (premature ventricular contractions)
            if abnormality_type == 'pvcs':
                if np.random.random() < 0.1:  # 10% chance of PVC
                    rr_interval *= 0.6  # Premature beat
                    
            beat_times.append(current_time)
            current_time += rr_interval
        
        # Generate ECG morphology
        for i, beat_time in enumerate(beat_times):
            if beat_time + 0.4 < duration:
                # Check if this is a PVC
                is_pvc = False
                if abnormality_type == 'pvcs' and i > 0:
                    prev_interval = beat_time - beat_times[i-1]
                    expected_interval = 60 / base_hr
                    if prev_interval < 0.8 * expected_interval:
                        is_pvc = True
                
                if is_pvc:
                    # Generate wider, different morphology QRS for PVC
                    self._add_pvc_complex(ecg, t, beat_time)
                else:
                    # Normal QRS complex
                    self._add_normal_complex(ecg, t, beat_time)
        
        # Add noise
        noise = np.random.normal(0, 0.02, len(ecg))
        ecg_noisy = ecg + noise
        
        return {
            'signal': ecg_noisy,
            'sampling_rate': self.fs,
            'time': t,
            'abnormality_type': abnormality_type,
            'severity': severity,
            'true_hr': base_hr,
            'beat_times': beat_times
        }
    
    def _add_normal_complex(self, ecg, t, beat_time):
        """Add normal PQRST complex"""
        # P wave
        p_start = beat_time
        p_duration = 0.08
        p_indices = np.where((t >= p_start) & (t <= p_start + p_duration))[0]
        if len(p_indices) > 0:
            p_wave = 0.1 * np.sin(np.pi * (t[p_indices] - p_start) / p_duration)
            ecg[p_indices] += p_wave
        
        # QRS complex
        qrs_start = beat_time + 0.12
        qrs_duration = 0.08
        qrs_indices = np.where((t >= qrs_start) & (t <= qrs_start + qrs_duration))[0]
        if len(qrs_indices) > 0:
            qrs_t = (t[qrs_indices] - qrs_start) / qrs_duration
            r_wave = 1.0 * np.exp(-((qrs_t - 0.5) / 0.15) ** 2)
            ecg[qrs_indices] += r_wave
        
        # T wave
        t_start = qrs_start + qrs_duration + 0.05
        t_duration = 0.16
        t_indices = np.where((t >= t_start) & (t <= t_start + t_duration))[0]
        if len(t_indices) > 0:
            t_wave = 0.3 * np.sin(np.pi * (t[t_indices] - t_start) / t_duration)
            ecg[t_indices] += t_wave
    
    def _add_pvc_complex(self, ecg, t, beat_time):
        """Add PVC (wider, different morphology)"""
        # No P wave for PVC
        
        # Wider QRS complex
        qrs_start = beat_time
        qrs_duration = 0.12  # Wider than normal
        qrs_indices = np.where((t >= qrs_start) & (t <= qrs_start + qrs_duration))[0]
        if len(qrs_indices) > 0:
            qrs_t = (t[qrs_indices] - qrs_start) / qrs_duration
            # Different morphology - more irregular
            r_wave = -0.8 * np.exp(-((qrs_t - 0.3) / 0.2) ** 2)  # Negative deflection
            s_wave = 0.6 * np.exp(-((qrs_t - 0.7) / 0.2) ** 2)   # Positive deflection
            ecg[qrs_indices] += r_wave + s_wave
        
        # Inverted T wave
        t_start = qrs_start + qrs_duration + 0.02
        t_duration = 0.16
        t_indices = np.where((t >= t_start) & (t <= t_start + t_duration))[0]
        if len(t_indices) > 0:
            t_wave = -0.2 * np.sin(np.pi * (t[t_indices] - t_start) / t_duration)