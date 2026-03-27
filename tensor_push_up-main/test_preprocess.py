"""
Simple test for preprocessing functionality
"""
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import cv2
import numpy as np

# Test imports
try:
    from src.pose_estimator import PoseEstimator
    print("[OK] PoseEstimator imported successfully")
except Exception as e:
    print(f"[FAIL] Failed to import PoseEstimator: {e}")
    sys.exit(1)

# Test pose estimation
print("\nTesting pose estimation...")
try:
    # Create a simple test frame
    test_frame = np.zeros((480, 640, 3), dtype=np.uint8)
    test_frame[:] = [100, 100, 100]  # Gray background

    # Test with IMAGE mode (no timestamp needed)
    estimator = PoseEstimator(static_image_mode=True)
    print("[OK] PoseEstimator initialized with static_image_mode=True")

    # Process a frame
    keypoints, angles = estimator.process_frame(test_frame)

    if keypoints is not None:
        print(f"[OK] Pose detected - keypoints shape: {keypoints.shape}")
        print(f"[OK] Angles calculated: {list(angles.keys())[:3]}...")
    else:
        print("[INFO] No pose detected (this is expected for blank frame)")

except Exception as e:
    print(f"[FAIL] Error during pose estimation: {e}")
    import traceback
    traceback.print_exc()

# Test data saving
print("\nTesting data saving...")
test_dir = "test_output"
os.makedirs(test_dir, exist_ok=True)
test_data = np.random.rand(10, 50)
test_path = os.path.join(test_dir, "test_data.npy")
np.save(test_path, test_data)

if os.path.exists(test_path):
    loaded = np.load(test_path)
    print(f"[OK] Data saved and loaded successfully - shape: {loaded.shape}")
else:
    print("[FAIL] Failed to save or load data")

# Test configuration loading
print("\nTesting configuration loading...")
try:
    from src.utils import load_config
    config = load_config('configs/train.yaml')
    print(f"[OK] Configuration loaded - model: {config['model']['name']}")
except Exception as e:
    print(f"[FAIL] Failed to load configuration: {e}")

print("\n" + "="*50)
print("TEST SUMMARY")
print("="*50)
print("All basic functionality tests completed!")
print("="*50)
