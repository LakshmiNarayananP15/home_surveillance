import os
import time
import csv
from datetime import datetime
import cv2
import numpy as np
from ultralytics import YOLO
import insightface
from insightface.app import FaceAnalysis
from sklearn.metrics.pairwise import cosine_similarity
import onnxruntime as ort

# --- Configuration ---
KNOWN_FACES_DIR = "known_faces"
LOG_DIR = "logs"
LOG_FILE = os.path.join(LOG_DIR, "events.csv")
ANTI_SPOOF_MODEL_PATH = "2.7_80x80_MiniFASNetV2.onnx"  # Match exact downloaded filename
CONFIDENCE_THRESHOLD = 0.45  # Cosine similarity threshold for face match

def setup_directories():
    os.makedirs(KNOWN_FACES_DIR, exist_ok=True)
    os.makedirs(LOG_DIR, exist_ok=True)
    if not os.path.exists(LOG_FILE):
        with open(LOG_FILE, mode='w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(["Timestamp", "Event_Type", "Name_Or_ID"])

def log_event(event_type, name):
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    with open(LOG_FILE, mode='a', newline='') as f:
        writer = csv.writer(f)
        writer.writerow([timestamp, event_type, name])
    print(f"[{timestamp}] ALERT: {event_type} - {name}")

def load_known_faces(app):
    """Loads images from known_faces/ folder and extracts 512-d embeddings."""
    known_embeddings = []
    known_names = []

    if not os.listdir(KNOWN_FACES_DIR):
        print("[WARNING] 'known_faces/' folder is empty! All faces will be marked UNKNOWN.")
        return known_embeddings, known_names

    print("[INFO] Loading known faces database...")
    for filename in os.listdir(KNOWN_FACES_DIR):
        if filename.lower().endswith(('.jpg', '.jpeg', '.png')):
            name = os.path.splitext(filename)[0].replace("_", " ").title()
            path = os.path.join(KNOWN_FACES_DIR, filename)
            
            img = cv2.imread(path)
            if img is None:
                continue
            
            faces = app.get(img)
            if len(faces) > 0:
                embedding = faces[0].embedding
                known_embeddings.append(embedding)
                known_names.append(name)
                print(f" Loaded face profile: {name}")
            else:
                print(f"[WARNING] No clear face detected in reference image: {filename}")
                
    return np.array(known_embeddings), known_names


# --- Anti-Spoofing Class with 2.7x Bounding Box Expansion ---
class AntiSpoofDetector:
    def __init__(self, model_path):
        if not os.path.exists(model_path):
            alt_path = "MiniFASNetV2.onnx"
            if os.path.exists(alt_path):
                model_path = alt_path
            else:
                raise FileNotFoundError(f"[ERROR] Anti-spoofing model not found at {model_path} or {alt_path}.")
        
        self.session = ort.InferenceSession(model_path, providers=['CPUExecutionProvider'])
        self.input_name = self.session.get_inputs()[0].name
        self.scale = 2.7  # MiniFASNet requires a 2.7x expanded crop to analyze surrounding context

    def is_real(self, frame, bbox):
        """Returns True if the face is real, False if it's a photo/screen."""
        x1, y1, x2, y2 = bbox
        box_w = x2 - x1
        box_h = y2 - y1
        center_x = x1 + box_w / 2.0
        center_y = y1 + box_h / 2.0
        
        src_h, src_w = frame.shape[:2]
        
        # Expand bounding box by 2.7x factor
        new_w = box_w * self.scale
        new_h = box_h * self.scale
        
        new_x1 = max(0, int(center_x - new_w / 2.0))
        new_y1 = max(0, int(center_y - new_h / 2.0))
        new_x2 = min(src_w - 1, int(center_x + new_w / 2.0))
        new_y2 = min(src_h - 1, int(center_y + new_h / 2.0))
        
        # Crop expanded frame region
        cropped = frame[new_y1:new_y2, new_x1:new_x2]
        if cropped.size == 0:
            return False

        # 1. Resize to 80x80 required input size
        face_resized = cv2.resize(cropped, (80, 80))
        
        # 2. Transpose HWC -> CHW format
        face_transposed = np.transpose(face_resized, (2, 0, 1))
        input_data = np.expand_dims(face_transposed, axis=0).astype(np.float32)
        
        # 3. Run ONNX inference
        outputs = self.session.run(None, {self.input_name: input_data})
        
        # 4. Parse class prediction (Index 1 = Real, Index 0 = Spoof)
        prediction = np.argmax(outputs[0])
        return prediction == 1


def main():
    setup_directories()

    # 1. Initialize YOLO Nano for fast human/body detection
    print("[INFO] Initializing YOLO model...")
    yolo_model = YOLO("yolov8n.pt")  

    # 2. Initialize InsightFace for Face Recognition
    print("[INFO] Initializing InsightFace recognition pipeline...")
    face_app = FaceAnalysis(name="buffalo_s", providers=['CPUExecutionProvider'])
    face_app.prepare(ctx_id=0, det_size=(320, 320))

    # 3. Initialize Anti-Spoofing Model
    print("[INFO] Initializing Anti-Spoofing model...")
    try:
        spoof_detector = AntiSpoofDetector(ANTI_SPOOF_MODEL_PATH)
    except FileNotFoundError as e:
        print(e)
        return

    # Load Face Profiles
    known_embeddings, known_names = load_known_faces(face_app)

    # Open Webcam Stream
    cap = cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

    if not cap.isOpened():
        print("[ERROR] Could not open webcam.")
        return

    print("[INFO] Surveillance system active. Press 'q' to exit.")
    
    last_logged = {}
    COOLDOWN_SECONDS = 10 

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        # --- NEW: Flip the frame horizontally to fix the mirror effect ---
        frame = cv2.flip(frame, 1)

        # YOLO Human Detection (runs silently in background)
        _ = yolo_model(frame, classes=[0], verbose=False, conf=0.5)
        
        # InsightFace Detection
        faces = face_app.get(frame)

        # Process Detected Faces Only
        for face in faces:
            bbox = face.bbox.astype(int)
            
            # Anti-Spoofing verification using frame and face bounding box
            is_live = spoof_detector.is_real(frame, bbox)
            
            if not is_live:
                # Highlight fake faces/screens with red box
                cv2.rectangle(frame, (bbox[0], bbox[1]), (bbox[2], bbox[3]), (0, 0, 255), 2)
                cv2.putText(frame, "FAKE / SPOOF", (bbox[0], bbox[1] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
                
                current_time = time.time()
                if "Spoof" not in last_logged or (current_time - last_logged["Spoof"] > COOLDOWN_SECONDS):
                    log_event("SPOOF_ATTEMPT", "Fake Face Detected")
                    last_logged["Spoof"] = current_time
                    
                continue  # Skip identity extraction for fake faces

            # If real, run embedding match
            emb = face.embedding
            name = "Unknown"
            color = (0, 165, 255)  # Orange for unknown real face

            if len(known_embeddings) > 0:
                similarities = cosine_similarity([emb], known_embeddings)[0]
                best_match_idx = np.argmax(similarities)
                best_score = similarities[best_match_idx]

                if best_score > CONFIDENCE_THRESHOLD:
                    name = known_names[best_match_idx]
                    color = (0, 255, 0)  # Green for known person
            
            # Cooldown logging
            current_time = time.time()
            if name == "Unknown":
                if name not in last_logged or (current_time - last_logged[name] > COOLDOWN_SECONDS):
                    log_event("UNKNOWN_INTRUDER_ALERT", "Unknown Person")
                    last_logged[name] = current_time
            else:
                if name not in last_logged or (current_time - last_logged[name] > COOLDOWN_SECONDS):
                    log_event("MEMBER_DETECTED", name)
                    last_logged[name] = current_time

            # Draw green/orange box for validated face only
            cv2.rectangle(frame, (bbox[0], bbox[1]), (bbox[2], bbox[3]), color, 2)
            label = f"{name} (Real)"
            cv2.putText(frame, label, (bbox[0], bbox[1] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

        # Render output
        cv2.imshow("Home Surveillance - Press 'q' to Quit", frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()