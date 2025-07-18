#!/usr/bin/env python3
"""
Test wfdb version and parameter compatibility
"""

import sys

def test_wfdb_version():
    """Test wfdb version and compatibility"""
    
    try:
        import wfdb
        print(f"✅ wfdb version: {wfdb.__version__}")
        
        # Test parameter compatibility
        print("\n🔍 Testing parameter compatibility...")
        
        # Test with a small record first
        test_record = '100'
        
        try:
            # Try newer parameter name (pn_dir)
            print("   Testing with pn_dir parameter...")
            record = wfdb.rdrecord(test_record, pn_dir='mitdb')
            annotation = wfdb.rdann(test_record, 'atr', pn_dir='mitdb')
            print("   ✅ pn_dir parameter works!")
            param_to_use = 'pn_dir'
            
        except TypeError as e:
            print(f"   ❌ pn_dir failed: {e}")
            
            try:
                # Try older parameter name (pb_dir)
                print("   Testing with pb_dir parameter...")
                record = wfdb.rdrecord(test_record, pb_dir='mitdb')
                annotation = wfdb.rdann(test_record, 'atr', pb_dir='mitdb')
                print("   ✅ pb_dir parameter works!")
                param_to_use = 'pb_dir'
                
            except Exception as e2:
                print(f"   ❌ pb_dir also failed: {e2}")
                return None
        
        # Test successful loading
        if record and annotation:
            print(f"\n📊 Successfully loaded record {test_record}:")
            print(f"   Signal shape: {record.p_signal.shape}")
            print(f"   Sampling rate: {record.fs} Hz")
            print(f"   Duration: {len(record.p_signal)/record.fs:.1f} seconds")
            print(f"   Annotations: {len(annotation.symbol)} beats")
            
            # Show some annotation examples
            print(f"   First 10 annotations: {annotation.symbol[:10]}")
            
        return param_to_use
        
    except ImportError:
        print("❌ wfdb package not installed")
        print("💡 Install with: pip install wfdb")
        return None
    except Exception as e:
        print(f"❌ Error testing wfdb: {e}")
        return None

def create_fixed_function(param_name):
    """Create a fixed version of the load function"""
    
    fixed_code = f'''
def load_mitbih_record_fixed(record_name, database='mitdb'):
    """Fixed version of load_mitbih_record with correct parameter"""
    import wfdb
    import numpy as np
    
    try:
        print(f"📥 Downloading MIT-BIH record {{record_name}} from {{database}}...")
        
        # Use the correct parameter name
        record = wfdb.rdrecord(record_name, {param_name}=database)
        annotation = wfdb.rdann(record_name, 'atr', {param_name}=database)
        
        # Extract ECG signal
        ecg_signal = record.p_signal[:, 0]
        
        # Process annotations
        annotation_map = {{
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
        }}
        
        processed_annotations = []
        for symbol, location in zip(annotation.symbol, annotation.sample):
            processed_annotations.append({{
                'sample': location,
                'time': location / record.fs,
                'symbol': symbol,
                'description': annotation_map.get(symbol, f'Unknown ({{symbol}})'),
                'is_abnormal': symbol not in ['N', '.', '+']
            }})
        
        return {{
            'signal': ecg_signal,
            'sampling_rate': record.fs,
            'time': np.arange(len(ecg_signal)) / record.fs,
            'annotations': processed_annotations,
            'record_name': record_name,
            'database': database,
            'duration': len(ecg_signal) / record.fs,
            'patient_info': {{
                'record': record_name,
                'leads': record.sig_name,
                'units': record.units
            }}
        }}
        
    except Exception as e:
        print(f"❌ Error loading record {{record_name}}: {{e}}")
        return None
'''
    
    return fixed_code

def main():
    """Main function"""
    print("🔍 wfdb Version and Compatibility Test")
    print("=" * 50)
    
    param_to_use = test_wfdb_version()
    
    if param_to_use:
        print(f"\n✅ Correct parameter to use: {param_to_use}")
        
        # Create fixed function
        fixed_code = create_fixed_function(param_to_use)
        
        # Save to file
        with open('wfdb_fixed_function.py', 'w') as f:
            f.write('# Fixed wfdb loading function\n')
            f.write(fixed_code)
        
        print(f"\n💾 Fixed function saved to: wfdb_fixed_function.py")
        print(f"💡 You can copy this function to replace the broken one in preprocessing.py")
        
        # Test the fixed function
        print(f"\n🧪 Testing fixed function...")
        try:
            exec(fixed_code)
            result = locals()['load_mitbih_record_fixed']('106', 'mitdb')
            
            if result:
                print(f"✅ Fixed function works!")
                print(f"   Loaded record 106 successfully")
                print(f"   Duration: {result['duration']:.1f} seconds")
                print(f"   Annotations: {len(result['annotations'])} beats")
                
                # Count abnormal beats
                abnormal_count = sum(1 for ann in result['annotations'] if ann['is_abnormal'])
                normal_count = len(result['annotations']) - abnormal_count
                print(f"   Normal beats: {normal_count}")
                print(f"   Abnormal beats: {abnormal_count}")
                
                return True
            else:
                print(f"❌ Fixed function failed")
                return False
                
        except Exception as e:
            print(f"❌ Error testing fixed function: {e}")
            return False
    else:
        print(f"\n❌ Could not determine correct parameter")
        return False

if __name__ == "__main__":
    success = main()
    
    if success:
        print(f"\n🎉 All tests passed!")
        print(f"💡 Your wfdb installation is working correctly")
        print(f"💡 Use the fixed function in preprocessing.py")
    else:
        print(f"\n❌ Tests failed")
        print(f"💡 Try reinstalling wfdb: pip install --upgrade wfdb")