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

# --- Configuration ---
KNOWN_FACES_DIR = "known_faces"
LOG_DIR = "logs"
LOG_FILE = os.path.join(LOG_DIR, "events.csv")
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
                # Take the embedding of the largest face found in the reference image
                embedding = faces[0].embedding
                known_embeddings.append(embedding)
                known_names.append(name)
                print(f" Loaded face profile: {name}")
            else:
                print(f"[WARNING] No clear face detected in reference image: {filename}")
                
    return np.array(known_embeddings), known_names

def main():
    setup_directories()

    # 1. Initialize YOLO Nano for fast human/body detection (CPU/Low-end friendly)
    print("[INFO] Initializing YOLO model...")
    yolo_model = YOLO("yolov8n.pt")  # You can swap with 'yolo26n.pt' if using Ultralytics latest

    # 2. Initialize InsightFace for Face Recognition (Using lightweight 'buffalo_s' pack)
    print("[INFO] Initializing InsightFace recognition pipeline...")
    face_app = FaceAnalysis(name="buffalo_s", providers=['CPUExecutionProvider'])
    face_app.prepare(ctx_id=0, det_size=(320, 320))  # 320x320 keeps CPU processing fast

    # Load Database
    known_embeddings, known_names = load_known_faces(face_app)

    # 3. Open Video Stream (0 for default USB webcam, or RTSP link for future IP cameras)
    cap = cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

    if not cap.isOpened():
        print("[ERROR] Could not open webcam.")
        return

    print("[INFO] Surveillance system active. Press 'q' to exit.")
    
    # Cooldown tracker to prevent spamming logs for the same person every frame
    last_logged = {}
    COOLDOWN_SECONDS = 10 

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        # Run YOLO object detection focusing on humans (class index 0)
        results = yolo_model(frame, classes=[0], verbose=False, conf=0.5)
        
        # Run InsightFace detection across the whole frame
        faces = face_app.get(frame)

        # Draw YOLO human bounding boxes
        for r in results:
            boxes = r.boxes
            for box in boxes:
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 0, 0), 2)
                cv2.putText(frame, "Human", (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)

        # Process Face Recognitions
        for face in faces:
            bbox = face.bbox.astype(int)
            emb = face.embedding
            
            name = "Unknown"
            color = (0, 0, 255) # Red for unknown

            if len(known_embeddings) > 0:
                # Compare current face embedding with database using cosine similarity
                similarities = cosine_similarity([emb], known_embeddings)[0]
                best_match_idx = np.argmax(similarities)
                best_score = similarities[best_match_idx]

                if best_score > CONFIDENCE_THRESHOLD:
                    name = known_names[best_match_idx]
                    color = (0, 255, 0) # Green for known family member
            
            # Handle logging cooldown
            current_time = time.time()
            if name == "Unknown":
                if name not in last_logged or (current_time - last_logged[name] > COOLDOWN_SECONDS):
                    log_event("UNKNOWN_INTRUDER_ALERT", "Unknown Person")
                    last_logged[name] = current_time
            else:
                if name not in last_logged or (current_time - last_logged[name] > COOLDOWN_SECONDS):
                    log_event("MEMBER_DETECTED", name)
                    last_logged[name] = current_time

            # Draw face bounding box and label
            cv2.rectangle(frame, (bbox[0], bbox[1]), (bbox[2], bbox[3]), color, 2)
            label = f"{name}"
            cv2.putText(frame, label, (bbox[0], bbox[1] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

        # Show live surveillance window
        cv2.imshow("Home Surveillance - Press 'q' to Quit", frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()