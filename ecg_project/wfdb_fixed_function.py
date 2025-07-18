# Fixed wfdb loading function

def load_mitbih_record_fixed(record_name, database='mitdb'):
    """Fixed version of load_mitbih_record with correct parameter"""
    import wfdb
    import numpy as np
    
    try:
        print(f"📥 Downloading MIT-BIH record {record_name} from {database}...")
        
        # Use the correct parameter name
        record = wfdb.rdrecord(record_name, pn_dir=database)
        annotation = wfdb.rdann(record_name, 'atr', pn_dir=database)
        
        # Extract ECG signal
        ecg_signal = record.p_signal[:, 0]
        
        # Process annotations
        annotation_map = {
            'N': 'Normal',
            'V': 'Premature ventricular contraction',
            'A': 'Atrial premature beat',
            'F': 'Fusion beat',
            'L': 'Left bundle branch block',
            'R': 'Right bundle branch block',
            'J': 'Nodal escape beat',
            'E': 'Ventricular escape beat',
            '/': 'Paced beat',
            'f': 'Fusion of paced and normal',
            'Q': 'Unclassifiable beat'
        }
        
        processed_annotations = []
        for symbol, location in zip(annotation.symbol, annotation.sample):
            processed_annotations.append({
                'sample': location,
                'time': location / record.fs,
                'symbol': symbol,
                'description': annotation_map.get(symbol, f'Unknown ({symbol})'),
                'is_abnormal': symbol not in ['N', '.', '+']
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
                'units': record.units
            }
        }
        
    except Exception as e:
        print(f"❌ Error loading record {record_name}: {e}")
        return None
