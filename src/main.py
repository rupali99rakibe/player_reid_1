# src/main.py
import os
import cv2
import numpy as np
from scipy.spatial.distance import cosine
from detector import PlayerDetector

def extract_embedding(crop):
    return crop.mean(axis=(0, 1))  # simple RGB vector average

def process_video(video_path, detector):
    cap = cv2.VideoCapture(video_path)
    all_detections = []
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        detections = detector.detect_players(frame)
        cropped_players = [frame[int(b[1]):int(b[3]), int(b[0]):int(b[2])] for b, _ in detections]
        embeddings = [extract_embedding(p) for p in cropped_players]
        all_detections.append((frame, detections, embeddings))
    cap.release()
    return all_detections

if __name__ == "__main__":
    detector = PlayerDetector("models/best.pt")

    print("Processing broadcast view...")
    broadcast_data = process_video("data/broadcast.mp4", detector)

    print("Processing tacticam view...")
    tacticam_data = process_video("data/tacticam.mp4", detector)

    # Step 1: Map tacticam players to broadcast IDs using embedding similarity
    broadcast_embeddings = broadcast_data[0][2]  # First frame embeddings
    tacticam_embeddings = tacticam_data[0][2]    # First frame embeddings

    mapping = {}
    for i, emb_tac in enumerate(tacticam_embeddings):
        sim_scores = [1 - cosine(emb_tac, emb_brd) for emb_brd in broadcast_embeddings]
        matched_idx = int(np.argmax(sim_scores))
        mapping[i] = matched_idx

    print("Player Mapping (Tacticam → Broadcast):", mapping)

    # Step 2: Write the result video
    os.makedirs("output", exist_ok=True)
    cap = cv2.VideoCapture("data/broadcast.mp4")
    fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    cap.release()

    out = cv2.VideoWriter("output/reid_result.mp4",
                          cv2.VideoWriter_fourcc(*'mp4v'),
                          fps,
                          (width, height))

    for i, (frame, detections, _) in enumerate(broadcast_data):
        for j, (bbox, conf) in enumerate(detections):
            x1, y1, x2, y2 = map(int, bbox)
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(frame, f"ID {j}", (x1, y1 - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        out.write(frame)

    out.release()
    print("✅ Output video saved to: output/reid_result.mp4")
