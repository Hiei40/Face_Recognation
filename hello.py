"""
Simple test script to verify OpenCV installation
"""
import sys

def test_opencv():
    try:
        import cv2
        print(f"[OK] OpenCV successfully installed!")
        print(f"  Version: {cv2.__version__}")
        print(f"  Python: {sys.version}")
        return True
    except ImportError as e:
        print(f"[ERROR] OpenCV not installed: {e}")
        print("  Install with: pip install opencv-python")
        return False

def test_camera():
    try:
        import cv2
        print("\n[OK] Testing camera access...")
        cap = cv2.VideoCapture(0)
        
        if not cap.isOpened():
            print("[ERROR] Cannot access camera")
            return False
        
        ret, frame = cap.read()
        cap.release()
        
        if ret:
            print(f"[OK] Camera working! Frame size: {frame.shape}")
            return True
        else:
            print("[ERROR] Cannot read from camera")
            return False
    except Exception as e:
        print(f"[ERROR] Camera test failed: {e}")
        return False

if __name__ == "__main__":
    print("=" * 50)
    print("OpenCV Installation Test")
    print("=" * 50)
    
    if test_opencv():
        test_camera()
    
    print("=" * 50)