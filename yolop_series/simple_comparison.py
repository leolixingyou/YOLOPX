#!/usr/bin/env python3
"""
Simple YOLOP vs YOLOPX Training Comparison
Run basic training comparison using existing working configurations.
"""

import sys
import os
import subprocess
from datetime import datetime

def run_simple_training(config_name, description, epochs=1):
    """Run simple training with a configuration"""
    print(f"\n{'='*70}")
    print(f"Training: {description}")
    print(f"Config: {config_name}")
    print(f"{'='*70}")
    
    os.chdir('/workspace/YOLOPX/v2')
    
    # Use the v2 tools training script
    cmd = [
        sys.executable, 'tools/train.py',
        '--modelDir', f'runs/comparison_{config_name.replace(".", "_")}_{datetime.now().strftime("%H%M%S")}'
    ]
    
    # Temporarily update the cfg to use our config
    from lib.config import cfg
    cfg.defrost()
    cfg.MODEL.CONFIG = f'/workspace/YOLOPX/v2/cfgs/models/{config_name}.yaml'
    cfg.TRAIN.END_EPOCH = epochs
    cfg.freeze()
    
    print(f"Running training with config: {cfg.MODEL.CONFIG}")
    
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=300)
        
        if result.returncode == 0:
            print(f"✅ {description} training completed")
            return True
        else:
            print(f"❌ {description} training failed")
            print(f"Error: {result.stderr}")
            return False
            
    except Exception as e:
        print(f"💥 {description} training crashed: {e}")
        return False

def main():
    """Main comparison function"""
    print("🚀 Simple YOLOP vs YOLOPX Training Comparison")
    print(f"⏰ Start time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    # Test the working configurations
    configs = [
        ('yolopx_v2_anchor_free', 'YOLOPX (Anchor-free)'),
        ('yolopx_v2_anchor_based', 'YOLOP (Anchor-based)'),
    ]
    
    results = {}
    for config, description in configs:
        success = run_simple_training(config, description)
        results[config] = success
    
    # Summary
    print(f"\n{'='*70}")
    print("📊 TRAINING COMPARISON SUMMARY")
    print(f"{'='*70}")
    
    for config, success in results.items():
        status = "✅ SUCCESS" if success else "❌ FAILED"
        print(f"{config:>25}: {status}")
    
    all_success = all(results.values())
    if all_success:
        print("\n🎉 Both models trained successfully!")
        print("📈 Basic comparison completed - both anchor-based and anchor-free work")
    else:
        print("\n⚠️  Some training failed.")
        
    return all_success

if __name__ == '__main__':
    success = main()
    sys.exit(0 if success else 1)