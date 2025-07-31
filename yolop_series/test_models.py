#!/usr/bin/env python3
"""
Quick test script to verify that both model configs can be loaded correctly.
"""

import sys
import os
import yaml
import torch

# Force the project root onto the Python path
sys.path.insert(0, '/workspace/YOLOPX')

try:
    from v2.models.builder import get_net_from_yaml
    print("✅ Successfully imported get_net_from_yaml")
except ImportError as e:
    print(f"❌ Failed to import get_net_from_yaml: {e}")
    sys.exit(1)

def test_model_config(model_name, config_path):
    """Test loading a model configuration"""
    print(f"\n{'='*60}")
    print(f"Testing {model_name.upper()} model configuration")
    print(f"Config path: {config_path}")
    print(f"{'='*60}")
    
    try:
        # Load the YAML config
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
        print("✅ Successfully loaded YAML config")
        
        # Try to create the model
        model = get_net_from_yaml(config_path)
        print("✅ Successfully created model")
        
        # Check model parameters
        total_params = sum(p.numel() for p in model.parameters())
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        
        print(f"📊 Model parameters:")
        print(f"   Total: {total_params:,}")
        print(f"   Trainable: {trainable_params:,}")
        
        # Test forward pass with dummy input
        model.eval()
        dummy_input = torch.randn(1, 3, 256, 256)
        
        with torch.no_grad():
            try:
                outputs = model(dummy_input)
                print("✅ Forward pass successful")
                print(f"📤 Output structure: {type(outputs)}")
                if isinstance(outputs, (list, tuple)):
                    print(f"   Number of outputs: {len(outputs)}")
                    for i, output in enumerate(outputs):
                        if hasattr(output, 'shape'):
                            print(f"   Output {i} shape: {output.shape}")
                elif hasattr(outputs, 'shape'):
                    print(f"   Output shape: {outputs.shape}")
                    
            except Exception as e:
                print(f"❌ Forward pass failed: {e}")
                return False
                
        return True
        
    except Exception as e:
        print(f"❌ Failed to test {model_name} config: {e}")
        return False

def main():
    """Main test function"""
    print("🔍 Testing YOLOP vs YOLOPX Model Configurations")
    print("="*80)
    
    models_to_test = [
        ('yolop', '/workspace/YOLOPX/v2/cfgs/models/yolop.yaml'),
        ('yolopx', '/workspace/YOLOPX/v2/cfgs/models/yolopx.yaml')
    ]
    
    results = {}
    
    for model_name, config_path in models_to_test:
        success = test_model_config(model_name, config_path)
        results[model_name] = success
    
    # Summary
    print(f"\n{'='*80}")
    print("📋 TEST SUMMARY")
    print(f"{'='*80}")
    
    all_passed = True
    for model_name, success in results.items():
        status = "✅ PASS" if success else "❌ FAIL"
        print(f"{model_name.upper():>8}: {status}")
        if not success:
            all_passed = False
    
    if all_passed:
        print("\n🎉 All model configurations are working correctly!")
        print("🚀 Ready to proceed with comparison experiments.")
    else:
        print("\n⚠️  Some model configurations failed. Please check the errors above.")
    
    return all_passed

if __name__ == '__main__':
    success = main()
    sys.exit(0 if success else 1)