"""
Configuration settings for Face Recognition System
Government Project - Customizable Security Levels
"""

import os
from pathlib import Path

# ============================================================================
# DIRECTORY SETTINGS
# ============================================================================

# Base directory for storing face data
KNOWN_FACES_DIR = Path("known_faces")
KNOWN_FACES_DIR.mkdir(exist_ok=True)

# Encodings file
ENCODINGS_FILE = KNOWN_FACES_DIR / "face_encodings.pkl"

# Logs directory (for audit trails)
LOGS_DIR = Path("logs")
LOGS_DIR.mkdir(exist_ok=True)

# ============================================================================
# FACE DETECTION SETTINGS
# ============================================================================

# Detection model: 'hog' (faster, CPU) or 'cnn' (more accurate, needs GPU)
# For government use: 'cnn' recommended if GPU available
# Detection model: 'hog' (faster, CPU) or 'cnn' (more accurate, needs GPU)
# For government use: 'cnn' recommended if GPU available
DETECTION_MODEL = 'hog' # Switched to HOG for speed on CPU
  # Change to 'cnn' for better accuracy with GPU

# Number of times to upsample image for detection (higher = detect smaller/obscured faces)
# Government recommendation: 1 for normal use, 2 for higher sensitivity
UPSAMPLE_TIMES = 1

# Number of jitters to use during encoding (higher = more accurate, but slower)
# 1 is standard, 10+ is high-accuracy for difficult cases
ENCODING_JITTERS = 1

# ============================================================================
# FACE RECOGNITION SETTINGS
# ============================================================================

# Security Level Presets
# STRICT: Government/High-Security (tolerance: 0.4)
# NORMAL: Standard Security (tolerance: 0.6)
# LENIENT: Accessibility Priority (tolerance: 0.72)

SECURITY_LEVELS = {
    'STRICT': {
        'tolerance': 0.4,
        'description': 'حكومي - أمان عالي جداً',
        'min_enrollments': 5,
    },
    'GOVERNMENT_ULTRA': {
        'tolerance': 0.78, # Extreme tolerance for single-feature recognition
        'description': 'حكومي - وضع الطوارئ (تعرف بأقل ملامح)',
        'min_enrollments': 10, # Requires many different occlusion states
    },
    'NORMAL': {
        'tolerance': 0.62,
        'description': 'عادي - متوازن',
        'min_enrollments': 3,
    },
    'LENIENT': {
        'tolerance': 0.72,
        'description': 'متساهل - سهولة الوصول',
        'min_enrollments': 2,
    }
}

# Current security level (change as needed)
CURRENT_SECURITY_LEVEL = 'GOVERNMENT_ULTRA'

# Get current tolerance
TOLERANCE = SECURITY_LEVELS[CURRENT_SECURITY_LEVEL]['tolerance']

# Minimum number of enrollment photos required
MIN_ENROLLMENTS = SECURITY_LEVELS[CURRENT_SECURITY_LEVEL]['min_enrollments']

# ============================================================================
# PERFORMANCE SETTINGS
# ============================================================================

# Process every Nth frame for faster recognition (1 = process all frames)
# Government recommendation: 1 for security, 2-3 for demo/testing
FRAME_SKIP = 3

# Downscale frame for faster detection (0.5 = half size, 1.0 = full size)
# Note: Increasing this improves detection of obscured faces
DETECTION_SCALE = 0.5 

# Maximum number of faces to process per frame
MAX_FACES_PER_FRAME = 5

# ============================================================================
# DISPLAY SETTINGS
# ============================================================================

# Show confidence percentage on screen
SHOW_CONFIDENCE = True

# Display frame rate (FPS) on screen
SHOW_FPS = True

# Use Arabic text in UI (requires Arabic font support)
USE_ARABIC_UI = True

# Color scheme
COLOR_RECOGNIZED = (0, 255, 0)    # Green
COLOR_UNKNOWN = (0, 0, 255)        # Red
COLOR_TEXT = (255, 255, 255)       # White

# ============================================================================
# LOGGING SETTINGS (For Government Audit)
# ============================================================================

# Enable logging of recognition events
ENABLE_LOGGING = True

# Log file path
LOG_FILE = LOGS_DIR / "recognition_log.txt"

# What to log
LOG_ENROLLMENTS = True      # Log when new faces are enrolled
LOG_RECOGNITIONS = True     # Log successful recognitions
LOG_UNKNOWNS = True         # Log unknown face detections
LOG_TIMESTAMPS = True       # Include timestamps in logs

# ============================================================================
# RECOGNITION PERSISTENCE (Memory)
# ============================================================================

# Remember recognized people for N frames even if recognition fails briefly
# This helps when person covers their face or moves quickly
RECOGNITION_PERSISTENCE = True
PERSISTENCE_FRAMES = 15  # About 0.5 - 1.0 second of memory

# ============================================================================
# ANTI-SPOOFING SETTINGS (Basic Liveness Detection)
# ============================================================================

# Enable basic anti-spoofing checks
ENABLE_ANTI_SPOOFING = False  # Set to True for production

# Require eye blink detection (experimental)
REQUIRE_BLINK = False

# ============================================================================
# WEBCAM SETTINGS
# ============================================================================

# Webcam device index (0 = default camera)
CAMERA_INDEX = 0

# Camera resolution (width, height)
# Higher = better quality but slower
CAMERA_RESOLUTION = (640, 480)

# Mirror webcam feed
MIRROR_CAMERA = True
