#!/usr/bin/env python3
"""
Real ECG Data Testing Script
============================

This script tests the real ECG data analysis capabilities by running
analysis on multiple MIT-BIH records with known conditions.

Usage:
    python test_real_ecg.py                    # Test all records
    python test_real_ecg.py --quick           # Test only featured records
    python test_real_ecg.py --record 100      # Test specific record
    python test_real_ecg.py --validate        # Validate against expert labels

Author: ECG Processing System
Version: 1.0
"""

import sys
import os
import argparse
import time
from datetime import datetime

# Add src directory to path
sys.path.append('src')

def test_single_record(record_name, database='mitdb', show_details=True):
    """Test a single ECG record"""
    
    try:
        from main import analyze_real_ecg_data
        
        print(f"\n🔍 Testing record {record_name}")
        print("-" * 40)
        
        start_time = time.time()
        results = analyze_real_ecg_data(record_name, database, 'results')
        end_time = time.time()
        
        if results:
            print(f"✅ Analysis completed in {end_time - start_time:.2f} seconds")
            
            # Show basic results
            report = results['report']
            print(f"   Duration: {report['signal_length']:.1f} seconds")
            print(f"   Detected beats: {report['num_beats']}")
            print(f"   Heart rate: {report['heart_rate']['mean']:.1f} bpm")
            print(f"   Our classification: {report['rhythm_classification']}")
            
            if report['abnormalities']:
                print(f"   Our detected abnormalities:")
                for abnormality in report['abnormalities']:
                    print(f"     • {abnormality}")
            else:
                print(f"   No abnormalities detected by our system")
            
            # Show expert comparison if available
            if 'expert_summary' in results:
                expert = results['expert_summary']
                record_info = results['record_info']
                
                print(f"   Known condition: {record_info['description']}")
                print(f"   Expert opinion: {expert['abnormal_percentage']:.1f}% abnormal beats")
                
                if show_details and expert['abnormal_beats'] > 0:
                    print(f"   Expert annotations:")
                    for ann_type, count in sorted(expert['annotation_types'].items()):
                        if count > 0:
                            print(f"     - {ann_type}: {count} beats")
                
                # Simple validation
                our_has_abnormal = len(report['abnormalities']) > 0
                expert_has_abnormal = expert['abnormal_percentage'] > 10  # 10% threshold
                
                if our_has_abnormal == expert_has_abnormal:
                    print(f"   ✅ Agreement: Both {'detect' if our_has_abnormal else 'do not detect'} abnormalities")
                else:
                    print(f"   ⚠️  Disagreement: Our system {'detects' if our_has_abnormal else 'does not detect'} abnormalities")
                    print(f"      Expert labels {expert['abnormal_percentage']:.1f}% as abnormal")
            
            return {
                'record': record_name,
                'success': True,
                'duration': end_time - start_time,
                'results': results
            }
            
        else:
            print(f"❌ Analysis failed for record {record_name}")
            return {
                'record': record_name,
                'success': False,
                'duration': end_time - start_time,
                'results': None
            }
            
    except Exception as e:
        print(f"❌ Error analyzing record {record_name}: {e}")
        return {
            'record': record_name,
            'success': False,
            'duration': 0,
            'error': str(e)
        }

def test_featured_records():
    """Test featured records with known conditions"""
    
    print("🏥 Testing Featured MIT-BIH Records")
    print("=" * 50)
    
    # Featured records with known conditions
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
    
    test_results = []
    total_start_time = time.time()
    
    for record_name, description in featured_records:
        print(f"\n📋 Expected: {description}")
        result = test_single_record(record_name, 'mitdb', show_details=False)
        test_results.append(result)
        
        # Add a small delay to be nice to PhysioNet servers
        time.sleep(1)
    
    total_end_time = time.time()
    
    # Summary
    print(f"\n📊 TESTING SUMMARY")
    print("=" * 50)
    successful_tests = sum(1 for r in test_results if r['success'])
    total_tests = len(test_results)
    
    print(f"✅ Successful analyses: {successful_tests}/{total_tests}")
    print(f"⏱️  Total testing time: {total_end_time - total_start_time:.1f} seconds")
    print(f"📈 Average time per record: {(total_end_time - total_start_time)/total_tests:.1f} seconds")
    
    # Show failed tests
    failed_tests = [r for r in test_results if not r['success']]
    if failed_tests:
        print(f"\n❌ Failed tests:")
        for test in failed_tests:
            print(f"   • Record {test['record']}: {test.get('error', 'Unknown error')}")
    
    return test_results

def test_all_records():
    """Test all available MIT-BIH records"""
    
    print("🏥 Testing All Available MIT-BIH Records")
    print("=" * 50)
    print("⚠️  This will take a while and download a lot of data!")
    
    response = input("Continue? (y/N): ")
    if response.lower() != 'y':
        print("Test cancelled.")
        return
    
    try:
        from preprocessing import RealECGDataLoader
        
        loader = RealECGDataLoader()
        all_records = loader.get_available_records('mitdb')
        
        print(f"📋 Testing {len(all_records)} records...")
        
        test_results = []
        
        for i, record in enumerate(all_records):
            print(f"\n[{i+1}/{len(all_records)}] Testing record {record}")
            result = test_single_record(record, 'mitdb', show_details=False)
            test_results.append(result)
            
            # Add delay between requests
            time.sleep(2)
        
        # Generate comprehensive report
        successful_tests = sum(1 for r in test_results if r['success'])
        print(f"\n📊 COMPREHENSIVE TESTING COMPLETE")
        print(f"✅ Successful: {successful_tests}/{len(all_records)} records")
        
        return test_results
        
    except Exception as e:
        print(f"❌ Error during comprehensive testing: {e}")
        return []

def validate_against_experts(test_results):
    """Validate our results against expert annotations"""
    
    print("\n👨‍⚕️ VALIDATION AGAINST EXPERT ANNOTATIONS")
    print("=" * 50)
    
    agreements = 0
    disagreements = 0
    no_expert_data = 0
    
    for result in test_results:
        if not result['success']:
            continue
            
        record = result['record']
        analysis = result['results']
        
        if 'expert_summary' not in analysis:
            no_expert_data += 1
            continue
        
        # Simple validation logic
        our_abnormal = len(analysis['report']['abnormalities']) > 0
        expert_abnormal = analysis['expert_summary']['abnormal_percentage'] > 10
        
        if our_abnormal == expert_abnormal:
            agreements += 1
            status = "✅ AGREE"
        else:
            disagreements += 1
            status = "❌ DISAGREE"
        
        print(f"   Record {record}: {status}")
        print(f"     Our system: {'Abnormal' if our_abnormal else 'Normal'}")
        print(f"     Expert: {analysis['expert_summary']['abnormal_percentage']:.1f}% abnormal")
    
    total_validated = agreements + disagreements
    if total_validated > 0:
        accuracy = (agreements / total_validated) * 100
        print(f"\n📊 VALIDATION SUMMARY:")
        print(f"   Agreement: {agreements}/{total_validated} ({accuracy:.1f}%)")
        print(f"   Disagreement: {disagreements}/{total_validated}")
        print(f"   No expert data: {no_expert_data}")
        
        if accuracy > 80:
            print(f"   🎉 Good agreement with expert annotations!")
        elif accuracy > 60:
            print(f"   ⚠️  Moderate agreement - consider tuning parameters")
        else:
            print(f"   ❌ Poor agreement - algorithm needs improvement")
    
    return {
        'agreements': agreements,
        'disagreements': disagreements,
        'accuracy': accuracy if total_validated > 0 else 0,
        'total_validated': total_validated
    }

def main():
    """Main testing function"""
    
    parser = argparse.ArgumentParser(description='Real ECG Data Testing')
    parser.add_argument('--quick', action='store_true', 
                       help='Test only featured records')
    parser.add_argument('--record', type=str, 
                       help='Test specific record')
    parser.add_argument('--validate', action='store_true',
                       help='Validate against expert annotations')
    parser.add_argument('--database', type=str, default='mitdb',
                       help='Database to test (mitdb, afdb)')
    
    args = parser.parse_args()
    
    print("🫀 Real ECG Data Testing System")
    print("=" * 50)
    print(f"Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    # Check if wfdb is available
    try:
        import wfdb
        print("✅ wfdb package is available")
    except ImportError:
        print("❌ wfdb package not found")
        print("💡 Install with: pip install wfdb")
        return
    
    test_results = []
    
    if args.record:
        # Test specific record
        test_results = [test_single_record(args.record, args.database)]
        
    elif args.quick:
        # Test featured records
        test_results = test_featured_records()
        
    else:
        # Test all records
        test_results = test_all_records()
    
    # Validate against experts if requested
    if args.validate and test_results:
        validate_against_experts(test_results)
    
    print(f"\n🎉 Testing completed!")
    print(f"Ended: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

if __name__ == "__main__":
    main()