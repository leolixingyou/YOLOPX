#!/usr/bin/env python3
"""Quick script to test single model training."""

import sys
import subprocess

if __name__ == '__main__':
    # Test with YOLOPx for 1 epoch
    cmd = [
        'python', 'train_unified.py',
        '--model', 'yolopx',
        '--epochs', '1',
        '--batch-size', '4',
        '--lr', '0.001',
        '--device', 'cuda:0',
        '--log-interval', '5'
    ]
    
    print("Running single model test...")
    print(f"Command: {' '.join(cmd)}")
    
    # Run the command
    result = subprocess.run(cmd)
    
    if result.returncode == 0:
        print("\n✓ Test successful!")
    else:
        print("\n✗ Test failed!")
        sys.exit(1)