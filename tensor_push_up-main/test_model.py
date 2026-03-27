"""
Test model creation
"""
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np
import tensorflow as tf
from tensorflow import keras

# Test imports
try:
    from src.model import create_model, compile_model
    print("[OK] Model module imported successfully")
except Exception as e:
    print(f"[FAIL] Failed to import model module: {e}")
    sys.exit(1)

print("\n" + "="*50)
print("Testing Model Creation")
print("="*50)

# Test 1: Create LSTM model
print("\n[TEST 1] Creating LSTM model...")
try:
    model = create_model(
        input_shape=(30, 50),
        num_classes=3,
        model_type='lstm',
        lstm_units=[128, 64],
        dense_units=[32, 16],
        dropout_rate=0.5
    )
    print(f"[OK] Model created: {model.__class__.__name__}")
    print(f"[OK] Input shape: {model.input_shape}")

    # Test 2: Compile model
    print("\n[TEST 2] Compiling model...")
    model = compile_model(model, learning_rate=0.001)
    print(f"[OK] Model compiled")

    # Test 3: Try prediction
    print("\n[TEST 3] Testing prediction...")
    test_input = np.random.random((1, 30, 50))
    output = model.predict(test_input, verbose=0)
    print(f"[OK] Prediction output shape: {output.shape}")

    print("\n" + "="*50)
    print("All tests passed!")
    print("="*50)

except Exception as e:
    print(f"[FAIL] Error during model creation: {e}")
    import traceback
    traceback.print_exc()
