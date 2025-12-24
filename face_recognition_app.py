"""
Advanced Face Recognition System
Government Project - High Accuracy Edition

Uses face_recognition library (dlib-based) for 99.38% accuracy
Supports partial face occlusion, varying lighting, and multiple angles
Perfect for government and security applications
"""

import cv2
import face_recognition
import numpy as np
import pickle
import argparse
import time
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Tuple, Optional
import threading

try:
    import pyttsx3
except ImportError:
    pyttsx3 = None
    print("⚠️ Warning: pyttsx3 not installed. Audio feedback disabled.")

# Import configuration
try:
    import config
except ImportError:
    print("⚠️ Warning: config.py not found. Using default settings.")
    # Fallback defaults
    class config:
        KNOWN_FACES_DIR = Path("known_faces")
        ENCODINGS_FILE = KNOWN_FACES_DIR / "face_encodings.pkl"
        LOGS_DIR = Path("logs")
        LOG_FILE = LOGS_DIR / "recognition_log.txt"
        DETECTION_MODEL = 'hog'
        UPSAMPLE_TIMES = 1
        TOLERANCE = 0.6
        MIN_ENROLLMENTS = 1
        FRAME_SKIP = 1
        DETECTION_SCALE = 0.5
        ENCODING_JITTERS = 1
        SHOW_CONFIDENCE = True
        SHOW_FPS = True
        COLOR_RECOGNIZED = (0, 255, 0)
        COLOR_UNKNOWN = (0, 0, 255)
        COLOR_TEXT = (255, 255, 255)
        ENABLE_LOGGING = True
        CAMERA_INDEX = 0
        CAMERA_RESOLUTION = (640, 480)
        MIRROR_CAMERA = True
        
        @staticmethod
        def __getattribute__(name):
            # Create directories if they don't exist
            if name in ['KNOWN_FACES_DIR', 'LOGS_DIR']:
                path = object.__getattribute__(config, name)
                path.mkdir(exist_ok=True)
                return path
            return object.__getattribute__(config, name)


class AdvancedFaceRecognition:
    """
    Advanced Face Recognition System using dlib-based face_recognition library
    
    Features:
    - 99.38% accuracy on LFW benchmark
    - Handles partial face occlusion (covered eyes, masks, etc.)
    - Robust to lighting changes and angles
    - Multiple enrollment photos per person
    - Recognition persistence (memory)
    - Logging and audit trail
    - Configurable security levels
    """
    
    def __init__(self):
        """Initialize the face recognition system"""
        # Create necessary directories
        config.KNOWN_FACES_DIR.mkdir(exist_ok=True)
        config.LOGS_DIR.mkdir(exist_ok=True)
        
        # Storage for face encodings
        # Format: {name: [encoding1, encoding2, ...]}
        self.known_face_encodings: Dict[str, List[np.ndarray]] = {}
        
        # Load existing encodings
        self.load_encodings()
        
        # Frame processing counter
        self.frame_count = 0
        
        # FPS calculation
        self.fps_start_time = time.time()
        self.fps_frame_count = 0
        self.current_fps = 0
        
        # Recognition persistence memory
        # Format: {face_index: {'name': name, 'confidence': conf, 'frames_left': N}}
        self.persistence_memory = {}
        
        self.log(f"System initialized with {len(self.known_face_encodings)} known identities")
        self.log(f"Detection Model: {config.DETECTION_MODEL.upper()}")
        self.log(f"Tolerance: {config.TOLERANCE}")

        # Initialize Text-to-Speech
        try:
            self.tts_engine = pyttsx3.init()
            self.tts_engine.setProperty('rate', 150) # Speed of speech
            self.last_speak_time = {} # Track last time name was spoken for cooldown
        except Exception as e:
            self.log(f"⚠️ Warning: Could not initialize TTS: {e}")
            self.tts_engine = None

    def speak_name(self, name: str):
        """Speak the name using TTS in a separate thread"""
        if not self.tts_engine:
            return

        current_time = time.time()
        
        # Cooldown check (don't repeat name too often - e.g. every 15 seconds)
        if name in self.last_speak_time:
            if current_time - self.last_speak_time[name] < 15:
                return
        
        self.last_speak_time[name] = current_time
        
        def run_speech():
            try:
                # Re-initialize engine for thread safety if needed, or use lock
                # For simplicity in this script, we'll try direct usage
                # Note: pyttsx3 can be finicky with threads. 
                # If it crashes, we might need a dedicated speech loop or queue.
                engine = pyttsx3.init() 
                engine.say(f"Welcome {name}")
                engine.runAndWait()
            except Exception as e:
                print(f"TTS Error: {e}")

        # Run in thread to not block video
        threading.Thread(target=run_speech, daemon=True).start()
    
    def log(self, message: str):
        """Log messages to file and console"""
        # Clean emojis for console if needed, but here we just use safe strings
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        log_message = f"[{timestamp}] {message}"
        
        # Print to console (safely)
        try:
            print(log_message)
        except UnicodeEncodeError:
            # Fallback for systems that don't support UTF-8 in console
            print(log_message.encode('ascii', errors='replace').decode('ascii'))
        
        # Write to log file if logging enabled
        if config.ENABLE_LOGGING:
            try:
                with open(config.LOG_FILE, 'a', encoding='utf-8') as f:
                    f.write(log_message + '\n')
            except Exception as e:
                print(f"Logging error: {e}")
    
    def load_encodings(self):
        """Load face encodings from disk"""
        if config.ENCODINGS_FILE.exists():
            try:
                with open(config.ENCODINGS_FILE, 'rb') as f:
                    self.known_face_encodings = pickle.load(f)
                
                total_encodings = sum(len(encodings) for encodings in self.known_face_encodings.values())
                print(f"Loaded {len(self.known_face_encodings)} identities ({total_encodings} total encodings)")
            except Exception as e:
                print(f"Error loading encodings: {e}")
                self.known_face_encodings = {}
        else:
            print("No existing face encodings found")
    
    def save_encodings(self):
        """Save face encodings to disk"""
        try:
            with open(config.ENCODINGS_FILE, 'wb') as f:
                pickle.dump(self.known_face_encodings, f)
            
            total_encodings = sum(len(encodings) for encodings in self.known_face_encodings.values())
            print(f"Saved {len(self.known_face_encodings)} identities ({total_encodings} total encodings)")
        except Exception as e:
            print(f"Error saving encodings: {e}")
    
    def enroll_face(self, name: str, num_photos: int = None):
        """
        Enroll a new face with multiple photos for better accuracy
        
        Args:
            name: Name of the person to enroll
            num_photos: Number of photos to capture (default: from config)
        """
        if num_photos is None:
            num_photos = config.MIN_ENROLLMENTS
        
        cap = cv2.VideoCapture(config.CAMERA_INDEX)
        
        if not cap.isOpened():
            self.log("❌ Error: Cannot access webcam")
            return False
        
        # Set camera resolution
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, config.CAMERA_RESOLUTION[0])
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, config.CAMERA_RESOLUTION[1])
        
        print(f"\n{'='*60}")
        print(f"ENROLLING: {name}")
        print(f"{'='*60}")
        print(f"Will capture {num_photos} photo(s) for best accuracy")
        print(f"")
        print(f"Instructions (Extreme Occlusion Mode):")
        print(f"  * Photo 1-2: Normal Face (Front/Side)")
        print(f"  * Photo 3-4: COVER EYES (Only nose/mouth visible)")
        print(f"  * Photo 5-6: COVER MOUTH (Only eyes/nose visible)")
        print(f"  * Photo 7-8: COVER LEFT/RIGHT half of face")
        print(f"  * Photo 9-10: Extreme angles or chin-up/down")
        print(f"  * Press SPACE to capture | Q to cancel")
        print(f"{'='*60}\n")
        
        captured_encodings = []
        captured_images = []
        photo_num = 1
        
        while len(captured_encodings) < num_photos:
            ret, frame = cap.read()
            if not ret:
                break
            
            # Mirror frame if configured
            if config.MIRROR_CAMERA:
                frame = cv2.flip(frame, 1)
            
            # Convert BGR to RGB for face_recognition
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            
            # Detect faces and landmarks
            face_locations = face_recognition.face_locations(
                rgb_frame,
                model=config.DETECTION_MODEL,
                number_of_times_to_upsample=config.UPSAMPLE_TIMES
            )
            
            # Get landmarks for visualization
            face_landmarks_list = face_recognition.face_landmarks(rgb_frame, face_locations)
            
            # Draw rectangles and landmarks
            display_frame = frame.copy()
            for (top, right, bottom, left), face_landmarks in zip(face_locations, face_landmarks_list):
                # Draw box
                cv2.rectangle(display_frame, (left, top), (right, bottom), (0, 255, 0), 2)
                
                # Draw landmarks (Feature dots)
                for feature, points in face_landmarks.items():
                    # Colors for different features: Nose(Blue), Mouth(Yellow), Chin(White)
                    color = (255, 0, 0) if feature == "nose_bridge" or feature == "nose_tip" else \
                            (0, 255, 255) if "mouth" in feature else (255, 255, 255)
                    
                    for (x, y) in points:
                        cv2.circle(display_frame, (x, y), 2, color, -1)
            
            # Display instructions
            cv2.putText(display_frame, f"Enrolling: {name}", (10, 30),
                       cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
            cv2.putText(display_frame, f"Photo {photo_num}/{num_photos}", (10, 70),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            
            if len(face_locations) > 0:
                cv2.putText(display_frame, "SPACE: Capture | Q: Cancel", (10, 110),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
            else:
                cv2.putText(display_frame, "No face detected - adjust position", (10, 110),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
            
            cv2.imshow('Face Enrollment', display_frame)
            
            key = cv2.waitKey(1) & 0xFF
            
            if key == ord('q'):
                print("❌ Enrollment cancelled")
                cap.release()
                cv2.destroyAllWindows()
                return False
            
            elif key == ord(' ') and len(face_locations) > 0:
                # Capture the face
                print(f"Capturing photo {photo_num}/{num_photos}...", end='')
                
                # Get face encoding with jitters for high precision
                face_encodings = face_recognition.face_encodings(
                    rgb_frame,
                    known_face_locations=face_locations,
                    num_jitters=config.ENCODING_JITTERS
                )
                
                if len(face_encodings) > 0:
                    captured_encodings.append(face_encodings[0])
                    captured_images.append(frame.copy())
                    print(" Success!")
                    photo_num += 1
                    time.sleep(0.5)  # Brief pause to avoid duplicate captures
                else:
                    print(" Failed to encode face, try again")
        
        cap.release()
        cv2.destroyAllWindows()
        
        # Save encodings
        if name in self.known_face_encodings:
            print(f"Adding {len(captured_encodings)} new photo(s) to existing enrollment")
            self.known_face_encodings[name].extend(captured_encodings)
        else:
            self.known_face_encodings[name] = captured_encodings
        
        self.save_encodings()
        
        # Save first captured image
        face_image_path = config.KNOWN_FACES_DIR / f"{name.replace(' ', '_')}.jpg"
        cv2.imwrite(str(face_image_path), captured_images[0])
        
        self.log(f"ENROLLED: {name} ({len(captured_encodings)} photos)")
        print(f"\n{'='*60}")
        print(f"Successfully enrolled {name}!")
        print(f"   Total photos: {len(self.known_face_encodings[name])}")
        print(f"{'='*60}\n")
        
        return True
    
    def recognize_faces(self):
        """Run real-time face recognition from webcam"""
        cap = cv2.VideoCapture(config.CAMERA_INDEX)
        
        if not cap.isOpened():
            self.log("Error: Cannot access webcam")
            return
        
        # Set camera resolution
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, config.CAMERA_RESOLUTION[0])
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, config.CAMERA_RESOLUTION[1])
        
        print(f"\n{'='*60}")
        print(f"FACE RECOGNITION ACTIVE")
        print(f"{'='*60}")
        print(f"Configuration:")
        print(f"  * Detection Model: {config.DETECTION_MODEL.upper()}")
        print(f"  * Tolerance: {config.TOLERANCE}")
        print(f"  * Known Identities: {len(self.known_face_encodings)}")
        print(f"\nPress 'q' to quit")
        print(f"{'='*60}\n")
        
        self.log("Recognition session started")
        
        # Variables to hold calculated results for skipped frames
        process_this_frame = True
        last_results = [] # Stores Tuples of (top, right, bottom, left, name, confidence, landmarks)
        
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            self.frame_count += 1
            
            # Mirror frame if configured
            if config.MIRROR_CAMERA:
                frame = cv2.flip(frame, 1)
            
            # Process every Nth frame for performance
            if self.frame_count % config.FRAME_SKIP == 0:
                process_this_frame = True
            else:
                process_this_frame = False
                
            if process_this_frame:
                # Downscale for faster detection
                small_frame = cv2.resize(frame, (0, 0), fx=config.DETECTION_SCALE, fy=config.DETECTION_SCALE)
                rgb_small_frame = cv2.cvtColor(small_frame, cv2.COLOR_BGR2RGB)
                
                # Detect faces and landmarks
                face_locations = face_recognition.face_locations(
                    rgb_small_frame,
                    model=config.DETECTION_MODEL,
                    number_of_times_to_upsample=config.UPSAMPLE_TIMES
                )
                
                # Get landmarks on small frame for speed
                face_landmarks_small = face_recognition.face_landmarks(rgb_small_frame, face_locations)
                
                # Scale back up face locations and landmarks
                scaled_locations = []
                scaled_landmarks_list = []
                
                for (top, right, bottom, left), landmarks in zip(face_locations, face_landmarks_small):
                    # Scale box
                    scaled_locations.append((
                        int(top/config.DETECTION_SCALE), 
                        int(right/config.DETECTION_SCALE),
                        int(bottom/config.DETECTION_SCALE), 
                        int(left/config.DETECTION_SCALE)
                    ))
                    
                    # Scale landmarks
                    scaled_landmarks = {}
                    for feature, points in landmarks.items():
                        scaled_landmarks[feature] = [
                            (int(x/config.DETECTION_SCALE), int(y/config.DETECTION_SCALE)) 
                            for (x, y) in points
                        ]
                    scaled_landmarks_list.append(scaled_landmarks)
                
                face_locations = scaled_locations
                
                # Get face encodings (use full resolution for accuracy)
                rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                face_encodings = face_recognition.face_encodings(rgb_frame, face_locations)
                
                # Clear previous results
                last_results = []
                
                # Track which known faces were seen this frame
                matched_names = set()

                # Recognize each face
                for (top, right, bottom, left), face_encoding, face_landmarks in zip(face_locations, face_encodings, scaled_landmarks_list):
                    # Compare with known faces
                    name = "Unknown"
                    confidence = 0
                    min_dist_found = 1.0 # Max distance
                    closest_person = "None"
                    
                    for known_name, known_encodings in self.known_face_encodings.items():
                        # Compare with all encodings for this person
                        distances = face_recognition.face_distance(known_encodings, face_encoding)
                        best_match_distance = np.min(distances)
                        
                        if best_match_distance < min_dist_found:
                            min_dist_found = best_match_distance
                            closest_person = known_name
                        
                        # Convert distance to confidence percentage
                        match_confidence = max(0, (1 - (best_match_distance / (config.TOLERANCE * 1.5))) * 100)
                        
                        if best_match_distance <= config.TOLERANCE and match_confidence > confidence:
                            name = known_name
                            confidence = match_confidence
                    
                    # Apply Persistence (Memory) - Link Unknowns to recent history
                    if config.RECOGNITION_PERSISTENCE:
                        if name == "Unknown":
                            # Look for the closest recent memory spatially
                            best_mem_dist = 1000
                            best_mem_data = None
                            
                            for old_id, data in self.persistence_memory.items():
                                if data['frames_left'] > 0:
                                    old_l, old_t = data['rect'][3], data['rect'][0] # left, top
                                    dist = np.sqrt((left - old_l)**2 + (top - old_t)**2)
                                    if dist < 150 and dist < best_mem_dist: # 150 pixels radius
                                        best_mem_dist = dist
                                        best_mem_data = data
                            
                            if best_mem_data:
                                name = best_mem_data['name'] + " (Memory)"
                                confidence = best_mem_data['confidence']
                                
                                # Use the memory name (minus suffix) to track persistence
                                real_name = best_mem_data['name']
                                matched_names.add(real_name)
                                
                                # Update memory with new position
                                self.persistence_memory[real_name] = {
                                    'name': real_name,
                                    'confidence': confidence,
                                    'frames_left': config.PERSISTENCE_FRAMES,
                                    'rect': (top, right, bottom, left),
                                    'landmarks': face_landmarks
                                }
                        else:
                            # Update/Create memory
                            matched_names.add(name)
                            self.persistence_memory[name] = {
                                'name': name,
                                'confidence': confidence,
                                'frames_left': config.PERSISTENCE_FRAMES,
                                'rect': (top, right, bottom, left),
                                'landmarks': face_landmarks
                            }
                    
                    # Store result for drawing this frame AND execution in skipped frames
                    last_results.append((top, right, bottom, left, name, confidence, face_landmarks))
                
                # Recover lost faces from memory (Anti-Flicker)
                if config.RECOGNITION_PERSISTENCE:
                    for mem_name, data in self.persistence_memory.items():
                        if mem_name not in matched_names and data['frames_left'] > 0:
                            # Verify if this matches "Unknown" from above? No, handled above.
                            # This block is for faces NOT detected at all by the model this frame
                            
                            # Decrement life
                            data['frames_left'] -= 1
                            
                            if data['frames_left'] > 0:
                                # Add to results to keep drawing it
                                # We use the OLD position (static)
                                # Ideally we could use a tracker, but static is fine for flicker fix
                                top, right, bottom, left = data['rect']
                                confidence = data['confidence']
                                landmarks = data['landmarks']
                                display_name = data['name'] + " (Holding)"
                                
                                last_results.append((top, right, bottom, left, display_name, confidence, landmarks))

            # DRAW RESULTS (Run every frame using last_results)
            for (top, right, bottom, left, name, confidence, face_landmarks) in last_results:

                if name != "Unknown":
                    self.speak_name(name.replace(" (Memory)", "").replace(" (Holding)", ""))

                # Expand bounding box to cover "ears" / head
                # Face detection usually gives a tight box. We'll make it wider and taller.
                box_width = right - left
                box_height = bottom - top
                
                # Expand by 20% on sides and 30% on top (for hair/head)
                expand_w = int(box_width * 0.2)
                expand_h_top = int(box_height * 0.3)
                
                # Calculate new coordinates with clamping to frame boundaries
                frame_h, frame_w = frame.shape[:2]
                
                disp_left = max(0, left - expand_w)
                disp_right = min(frame_w, right + expand_w)
                disp_top = max(0, top - expand_h_top)
                disp_bottom = bottom # Keep bottom same (chin usually okay)

                # Draw rectangle around face (Expanded)
                color = config.COLOR_RECOGNIZED if "Unknown" not in name else config.COLOR_UNKNOWN
                cv2.rectangle(frame, (disp_left, disp_top), (disp_right, disp_bottom), color, 2)
                
                # Draw label with name and confidence
                label = name
                if config.SHOW_CONFIDENCE and "Unknown" not in name:
                    label += f" ({confidence:.1f}%)"
                
                # Draw facial landmarks (Feature Focus Visualization)
                for feature, points in face_landmarks.items():
                    # Colors: Nose(Blue), Mouth(Yellow), Chin(White)
                    landmark_color = (255, 0, 0) if "nose" in feature else \
                                    (0, 255, 255) if "mouth" in feature else (200, 200, 200)
                    
                    # Use Red for eyes if Unknown (shows it's looking there even if failing)
                    if "eye" in feature and "Unknown" in name:
                        landmark_color = (0, 0, 255)
                        
                    for (lx, ly) in points:
                        cv2.circle(frame, (lx, ly), 1, landmark_color, -1)

                # Draw background for text
                cv2.rectangle(frame, (disp_left, disp_bottom - 35), (disp_right, disp_bottom), color, cv2.FILLED)
                cv2.putText(frame, label, (disp_left + 6, disp_bottom - 6),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
                
                # Log recognition with debug info for extreme cases
                if process_this_frame: # Only log on processed frames to avoid spam
                    if "Unknown" not in name:
                        self.log(f"RECOGNIZED: {name} (conf: {confidence:.1f}%)")
            
            # Calculate and display FPS
            self.fps_frame_count += 1
            if time.time() - self.fps_start_time > 1:
                self.current_fps = self.fps_frame_count / (time.time() - self.fps_start_time)
                self.fps_start_time = time.time()
                self.fps_frame_count = 0
            
            # Cleanup persistence memory (remove stale entries)
            if config.RECOGNITION_PERSISTENCE:
                self.persistence_memory = {fid: data for fid, data in self.persistence_memory.items() 
                                         if data['frames_left'] > 0}

            # Display info overlay
            cv2.putText(frame, f"Known: {len(self.known_face_encodings)}", (10, 30),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, config.COLOR_TEXT, 2)
            
            if config.SHOW_FPS:
                cv2.putText(frame, f"FPS: {self.current_fps:.1f}", (10, 60),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, config.COLOR_TEXT, 2)
            
            cv2.putText(frame, "Press 'q' to quit", (10, 90),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, config.COLOR_TEXT, 1)
            
            # Use English title to avoid potential display issues
            cv2.imshow('Face Recognition - Government System', frame)
            
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        
        cap.release()
        cv2.destroyAllWindows()
        self.log("Recognition session ended")
    
    def list_enrolled(self):
        """List all enrolled identities"""
        print(f"\n{'='*60}")
        print(f"ENROLLED IDENTITIES ({len(self.known_face_encodings)})")
        print(f"{'='*60}")
        
        if len(self.known_face_encodings) == 0:
            print("  (No enrolled faces)")
        else:
            for i, (name, encodings) in enumerate(self.known_face_encodings.items(), 1):
                print(f"  {i}. {name} ({len(encodings)} photo(s))")
        
        print(f"{'='*60}\n")
    
    def delete_enrollment(self, name: str):
        """Delete an enrolled face"""
        if name in self.known_face_encodings:
            del self.known_face_encodings[name]
            self.save_encodings()
            
            # Delete image file if exists
            image_path = config.KNOWN_FACES_DIR / f"{name.replace(' ', '_')}.jpg"
            if image_path.exists():
                image_path.unlink()
            
            self.log(f"DELETED: {name}")
            print(f"Deleted enrollment for: {name}")
            return True
        else:
            print(f"Error: '{name}' not found in enrolled faces")
            return False


def main():
    """Main application entry point"""
    parser = argparse.ArgumentParser(
        description='Advanced Face Recognition System - Government Project',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  py face_recognition_app.py                        # Run recognition
  py face_recognition_app.py --enroll "John Doe"    # Enroll new person
  py face_recognition_app.py --list                 # List enrolled faces
  py face_recognition_app.py --delete "John Doe"    # Delete enrollment
  py face_recognition_app.py --enroll "Jane" --photos 5  # Enroll with 5 photos
        """
    )
    
    parser.add_argument('--enroll', type=str, metavar='NAME',
                       help='Enroll a new face with the given name')
    parser.add_argument('--photos', type=int, metavar='N',
                       help='Number of photos to capture during enrollment (default: from config)')
    parser.add_argument('--list', action='store_true',
                       help='List all enrolled faces')
    parser.add_argument('--delete', type=str, metavar='NAME',
                       help='Delete an enrolled face')
    parser.add_argument('--config-info', action='store_true',
                       help='Show current configuration')
    
    args = parser.parse_args()
    
    # Print header
    print("\n" + "="*60)
    print("ADVANCED FACE RECOGNITION SYSTEM")
    print("   Government Project - High Accuracy Edition")
    print("   Powered by dlib (99.38% accuracy)")
    print("="*60)
    
    # Initialize system
    fr = AdvancedFaceRecognition()
    
    # Handle commands
    if args.config_info:
        # Show configuration
        print(f"\nConfiguration Info:")
        print(f"  * Detection Model: {config.DETECTION_MODEL.upper()}")
        print(f"  * Tolerance: {config.TOLERANCE}")
        print(f"  * Min Enrollments: {config.MIN_ENROLLMENTS}")
        print(f"  * Frame Skip: {config.FRAME_SKIP}")
        print(f"  * Logging: {'Enabled' if config.ENABLE_LOGGING else 'Disabled'}")
        print()
    
    elif args.enroll:
        # Enroll mode
        fr.enroll_face(args.enroll, num_photos=args.photos)
    
    elif args.list:
        # List enrolled faces
        fr.list_enrolled()
    
    elif args.delete:
        # Delete enrollment
        fr.delete_enrollment(args.delete)
    
    else:
        # Recognition mode
        if len(fr.known_face_encodings) == 0:
            print("\nWARNING: No faces enrolled yet!")
            print("\nTo enroll a face, run:")
            print('  py face_recognition_app.py --enroll "Your Name"\n')
        else:
            fr.recognize_faces()


if __name__ == "__main__":
    main()
