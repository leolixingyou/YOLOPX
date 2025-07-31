#!/usr/bin/env python3
"""
YOLOP vs YOLOPX Comparison Script for Stage 1
This script runs short training experiments to compare anchor-based (YOLOP) vs anchor-free (YOLOPX) detection heads.
"""

import subprocess
import sys
import time
import os
from datetime import datetime

def run_training(model_name, epochs=5, logdir_suffix=""):
    """Run training for a specific model"""
    print(f"\n{'='*60}")
    print(f"Starting training for {model_name.upper()}")
    print(f"{'='*60}")
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_dir = f"runs/comparison_{model_name}_{timestamp}{logdir_suffix}"
    
    # Ensure we're in the right directory
    os.chdir('/workspace/YOLOPX/v2')
    
    cmd = [
        sys.executable, 'train.py',
        '--model_name', model_name,
        '--logDir', log_dir
    ]
    
    print(f"Running command: {' '.join(cmd)}")
    
    try:
        # Run the training process
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=600)  # 10 minute timeout
        
        if result.returncode == 0:
            print(f"✅ {model_name.upper()} training completed successfully")
            print(f"📁 Logs saved to: {log_dir}")
        else:
            print(f"❌ {model_name.upper()} training failed")
            print(f"STDERR: {result.stderr}")
            
        return result.returncode == 0, log_dir
        
    except subprocess.TimeoutExpired:
        print(f"⏰ {model_name.upper()} training timed out after 10 minutes")
        return False, log_dir
    except Exception as e:
        print(f"💥 {model_name.upper()} training crashed: {str(e)}")
        return False, log_dir

def main():
    """Main comparison function"""
    print("🚀 Starting YOLOP vs YOLOPX Comparison Experiment")
    print("📋 Stage 1: Anchor-based vs Anchor-free Detection Head Comparison")
    print(f"⏰ Start time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    results = {}
    
    # Test both models
    models = ['yolop', 'yolopx']
    
    for model in models:
        success, log_dir = run_training(model)
        results[model] = {
            'success': success,
            'log_dir': log_dir
        }
        
        # Brief pause between experiments
        time.sleep(5)
    
    # Summary
    print(f"\n{'='*80}")
    print("📊 COMPARISON EXPERIMENT SUMMARY")
    print(f"{'='*80}")
    
    for model, result in results.items():
        status = "✅ SUCCESS" if result['success'] else "❌ FAILED"
        print(f"{model.upper():>8}: {status} | Log dir: {result['log_dir']}")
    
    # Overall success
    all_success = all(result['success'] for result in results.values())
    if all_success:
        print("\n🎉 All experiments completed successfully!")
        print("📈 Ready for performance analysis and documentation.")
    else:
        print("\n⚠️  Some experiments failed. Check the logs for details.")
    
    print(f"⏰ End time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    return all_success

if __name__ == '__main__':
    success = main()
    sys.exit(0 if success else 1)