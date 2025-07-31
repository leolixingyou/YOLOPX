#!/usr/bin/env python3
"""
YOLOP vs YOLOPX Conflict Detection Comparison Script
This script runs conflict detection experiments to compare anchor-based vs anchor-free detection heads.
"""

import subprocess
import sys
import time
import os
from datetime import datetime

def run_conflict_detection(config_name, method="original", epochs=1, logdir_suffix=""):
    """Run conflict detection training for a specific configuration"""
    print(f"\n{'='*80}")
    print(f"Starting conflict detection: {config_name.upper()} - {method.upper()}")
    print(f"{'='*80}")
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_dir = f"runs/conflict_{config_name}_{method}_{timestamp}{logdir_suffix}"
    
    # Ensure we're in the right directory
    os.chdir('/workspace/YOLOPX/v2')
    
    # Use the existing conflict detection script
    cmd = [
        sys.executable, 'tools/xy_conflict_detect_gemini.py',
        '--config', f'cfgs/models/{config_name}.yaml',
        '--method', method,
        '--epochs', str(epochs),
        '--logDir', log_dir
    ]
    
    print(f"Running command: {' '.join(cmd)}")
    
    try:
        # Run the conflict detection process
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=1200)  # 20 minute timeout
        
        if result.returncode == 0:
            print(f"✅ {config_name.upper()} - {method.upper()} completed successfully")
            print(f"📁 Logs saved to: {log_dir}")
        else:
            print(f"❌ {config_name.upper()} - {method.upper()} failed")
            print(f"STDOUT: {result.stdout}")
            print(f"STDERR: {result.stderr}")
            
        return result.returncode == 0, log_dir, result.stdout
        
    except subprocess.TimeoutExpired:
        print(f"⏰ {config_name.upper()} - {method.upper()} timed out after 20 minutes")
        return False, log_dir, ""
    except Exception as e:
        print(f"💥 {config_name.upper()} - {method.upper()} crashed: {str(e)}")
        return False, log_dir, ""

def main():
    """Main conflict detection comparison function"""
    print("🔍 Starting YOLOP vs YOLOPX Conflict Detection Comparison")
    print("📋 Anchor-based vs Anchor-free Detection Head Conflict Analysis")
    print(f"⏰ Start time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    results = {}
    
    # Test configurations to compare
    configs_to_test = [
        ('yolopx_v2_anchor_free', 'YOLOPX (Anchor-free)'),
        ('yolopx_v2_anchor_based', 'YOLOP (Anchor-based)'),
    ]
    
    # Test original method first
    method = "original"
    
    for config, description in configs_to_test:
        print(f"\n🧪 Testing {description}")
        success, log_dir, output = run_conflict_detection(config, method)
        results[f"{config}_{method}"] = {
            'success': success,
            'log_dir': log_dir,
            'output': output,
            'description': description
        }
        
        # Brief pause between experiments
        time.sleep(5)
    
    # Summary
    print(f"\n{'='*100}")
    print("📊 CONFLICT DETECTION COMPARISON SUMMARY")
    print(f"{'='*100}")
    
    for key, result in results.items():
        status = "✅ SUCCESS" if result['success'] else "❌ FAILED"
        print(f"{result['description']:>25}: {status}")
        print(f"{'':>25}  Log dir: {result['log_dir']}")
        if result['success'] and result['output']:
            # Try to extract conflict metrics from output
            lines = result['output'].split('\n')
            conflict_lines = [line for line in lines if 'conflict' in line.lower() or 'tci' in line.lower()]
            for line in conflict_lines[:3]:  # Show first 3 conflict-related lines
                print(f"{'':>25}  {line.strip()}")
        print()
    
    # Overall success
    all_success = all(result['success'] for result in results.values())
    if all_success:
        print("🎉 All conflict detection experiments completed successfully!")
        print("📈 Ready for conflict analysis and comparison.")
        print("\n📋 Key Findings:")
        print("   - Both anchor-based and anchor-free models tested")
        print("   - Conflict metrics calculated for multi-task learning")
        print("   - Logs available for detailed analysis")
    else:
        print("⚠️  Some experiments failed. Check the logs for details.")
    
    print(f"⏰ End time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    return all_success

if __name__ == '__main__':
    success = main()
    sys.exit(0 if success else 1)