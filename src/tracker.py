# src/tracker.py
import numpy as np
import cv2
from scipy.spatial.distance import cosine

class SimpleTracker:
    def __init__(self):
        self.next_id = 0
        self.tracks = {}

    def update(self, detections, embeddings):
        updated_tracks = {}
        for i, (bbox, emb) in enumerate(zip(detections, embeddings)):
            matched_id = None
            for track_id, track_data in self.tracks.items():
                prev_emb = track_data['embedding']
                if cosine(emb, prev_emb) < 0.5:  # Cosine similarity threshold
                    matched_id = track_id
                    break
            if matched_id is None:
                matched_id = self.next_id
                self.next_id += 1
            updated_tracks[matched_id] = {'bbox': bbox, 'embedding': emb}
        self.tracks = updated_tracks
        return self.tracks
