"""
Simple Face Recognition Application
Supports face detection, enrollment, and recognition using OpenCV
"""
import cv2
import numpy as np
import os
import pickle
import argparse
from pathlib import Path

class SimpleFaceRecognition:
    def __init__(self):
        self.known_faces_dir = Path("known_faces")
        self.known_faces_dir.mkdir(exist_ok=True)
        self.encodings_file = self.known_faces_dir / "encodings.pkl"
        
        # Load face detection model (Haar Cascade)
        cascade_path = cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
        self.face_cascade = cv2.CascadeClassifier(cascade_path)
        
        # Load known faces
        self.known_encodings = {}
        self.load_encodings()
    
    def load_encodings(self):
        """Load previously saved face encodings"""
        if self.encodings_file.exists():
            with open(self.encodings_file, 'rb') as f:
                self.known_encodings = pickle.load(f)
            print(f"Loaded {len(self.known_encodings)} known faces")
        else:
            print("No existing face encodings found")
    
    def save_encodings(self):
        """Save face encodings to disk"""
        with open(self.encodings_file, 'wb') as f:
            pickle.dump(self.known_encodings, f)
        print(f"Saved encodings for {len(self.known_encodings)} faces")
    
    def extract_features(self, face_img):
        """
        Extract features with AGGRESSIVE lighting normalization
        Multiple preprocessing steps to handle extreme lighting variations
        """
        # Resize to standard size
        face_resized = cv2.resize(face_img, (100, 100))
        
        # Convert to grayscale if needed
        if len(face_resized.shape) == 3:
            face_resized = cv2.cvtColor(face_resized, cv2.COLOR_BGR2GRAY)
        
        # Focus on center 70% of face (core facial features, avoiding hair/head edges)
        # This makes it more robust to head coverings
        h, w = face_resized.shape
        y_start, y_end = int(h * 0.15), int(h * 0.85)
        x_start, x_end = int(w * 0.15), int(w * 0.85)
        face_core = face_resized[y_start:y_end, x_start:x_end]
        
        # Step 1: Apply Gaussian blur to reduce noise
        face_blur = cv2.GaussianBlur(face_core, (5, 5), 0)
        
        # Step 2: Apply STRONG CLAHE (increased clipLimit and smaller tiles)
        # This AGGRESSIVELY normalizes lighting
        clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(4, 4))
        face_clahe = clahe.apply(face_blur)
        
        # Step 3: Apply global histogram equalization for additional normalization
        face_normalized = cv2.equalizeHist(face_clahe)
        
        # Step 4: Add edge detection to emphasize facial structure
        edges = cv2.Sobel(face_normalized, cv2.CV_64F, 1, 1, ksize=3)
        edges = np.abs(edges)
        edges = (edges / edges.max() * 255).astype(np.uint8)
        
        # Combine normalized face with edge features (80/20 blend for stability)
        combined = cv2.addWeighted(face_normalized, 0.8, edges, 0.2, 0)
        
        # Resize to standard feature size
        combined = cv2.resize(combined, (70, 70))
        
        # Normalize to [0, 1] range and flatten
        features = combined.flatten().astype(np.float32) / 255.0
        
        # Additional normalization: zero mean, unit variance
        features = (features - np.mean(features)) / (np.std(features) + 1e-6)
        
        return features
    
    def compare_faces(self, features1, features2, threshold=0.30):
        """
        Compare two face feature vectors
        Returns True if faces match (similarity above threshold)
        Threshold: 0.30 is lenient to handle distance variations and appearance changes
        """
        # Calculate normalized correlation coefficient
        similarity = np.corrcoef(features1, features2)[0, 1]
        return similarity > threshold, similarity
    
    def detect_faces(self, frame):
        """
        Detect faces in a frame
        Optimized to detect faces at various distances (close and far)
        """
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = self.face_cascade.detectMultiScale(
            gray,
            scaleFactor=1.05,      # Smaller steps = better detection at various scales
            minNeighbors=3,        # Lower = more lenient (detect more faces)
            minSize=(30, 30),      # Smaller minimum = detect farther faces
            maxSize=(300, 300)     # Limit maximum size for performance
        )
        return faces
    
    def enroll_face(self, name):
        """Enroll a new face by capturing from webcam"""
        cap = cv2.VideoCapture(0)
        
        if not cap.isOpened():
            print("Error: Cannot access webcam")
            return False
        
        print(f"\nEnrolling face for: {name}")
        print("Position your face in the frame and press SPACE to capture")
        print("Press 'q' to quit without enrolling")
        
        enrolled = False
        
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            # Flip frame for mirror effect
            frame = cv2.flip(frame, 1)
            
            # Detect faces
            faces = self.detect_faces(frame)
            
            # Draw rectangles around faces
            for (x, y, w, h) in faces:
                cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
                cv2.putText(frame, "Press SPACE to capture", (x, y-10),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
            
            # Display instructions
            cv2.putText(frame, f"Enrolling: {name}", (10, 30),
                       cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
            cv2.putText(frame, "SPACE: Capture | Q: Quit", (10, 60),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
            
            cv2.imshow('Face Enrollment', frame)
            
            key = cv2.waitKey(1) & 0xFF
            
            if key == ord('q'):
                print("Enrollment cancelled")
                break
            elif key == ord(' ') and len(faces) > 0:
                # Capture the first detected face
                (x, y, w, h) = faces[0]
                face_img = frame[y:y+h, x:x+w]
                
                # Extract features
                features = self.extract_features(face_img)
                
                # Save encoding
                self.known_encodings[name] = features
                self.save_encodings()
                
                # Save face image
                face_path = self.known_faces_dir / f"{name.replace(' ', '_')}.jpg"
                cv2.imwrite(str(face_path), face_img)
                
                print(f"[SUCCESS] Successfully enrolled {name}!")
                enrolled = True
                break
        
        cap.release()
        cv2.destroyAllWindows()
        return enrolled
    
    def recognize_faces(self):
        """Run face recognition on webcam feed"""
        cap = cv2.VideoCapture(0)
        
        if not cap.isOpened():
            print("Error: Cannot access webcam")
            return
        
        print("\nStarting face recognition...")
        print("Press 'q' to quit")
        
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            # Flip frame for mirror effect
            frame = cv2.flip(frame, 1)
            
            # Detect faces
            faces = self.detect_faces(frame)
            
            # Recognize each detected face
            for (x, y, w, h) in faces:
                face_img = frame[y:y+h, x:x+w]
                features = self.extract_features(face_img)
                
                # Compare with known faces
                best_match = "Unknown"
                best_similarity = 0
                
                for name, known_features in self.known_encodings.items():
                    matches, similarity = self.compare_faces(features, known_features)
                    if matches and similarity > best_similarity:
                        best_similarity = similarity
                        best_match = name
                
                # Draw rectangle and name
                color = (0, 255, 0) if best_match != "Unknown" else (0, 0, 255)
                cv2.rectangle(frame, (x, y), (x+w, y+h), color, 2)
                
                # Display name and confidence
                label = f"{best_match}"
                if best_match != "Unknown":
                    label += f" ({best_similarity:.2f})"
                
                cv2.putText(frame, label, (x, y-10),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
            
            # Display info
            cv2.putText(frame, f"Known faces: {len(self.known_encodings)}", (10, 30),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            cv2.putText(frame, "Press 'q' to quit", (10, 60),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
            
            cv2.imshow('Face Recognition', frame)
            
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        
        cap.release()
        cv2.destroyAllWindows()

def main():
    parser = argparse.ArgumentParser(description='Simple Face Recognition Application')
    parser.add_argument('--enroll', type=str, help='Enroll a new face with the given name')
    parser.add_argument('--list', action='store_true', help='List all enrolled faces')
    
    args = parser.parse_args()
    
    # Initialize face recognition
    fr = SimpleFaceRecognition()
    
    if args.enroll:
        # Enroll mode
        fr.enroll_face(args.enroll)
    elif args.list:
        # List enrolled faces
        print(f"\nEnrolled faces ({len(fr.known_encodings)}):")
        for name in fr.known_encodings.keys():
            print(f"  - {name}")
    else:
        # Recognition mode
        if len(fr.known_encodings) == 0:
            print("\n[WARNING] No faces enrolled yet!")
            print("Enroll a face first using: py face_recognition_app.py --enroll \"Your Name\"")
        else:
            fr.recognize_faces()

if __name__ == "__main__":
    main()
