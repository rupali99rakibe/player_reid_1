# 🎥 Player Re-Identification – Cross-Camera Mode

## 📌 Objective
This mode aims to map player identities between two different camera views: **broadcast** and **tacticam**. It helps identify the same players captured from different camera angles.

---

## 📂 Folder Structure

player_reid_1/
├── data/
│ ├── broadcast.mp4
│ └── tacticam.mp4
├── models/
│ └── best.pt
├── src/
│ ├── common/
│ │ ├── detector.py
│ │ ├── tracker.py
│ │ └── utils.py
│ └── cross_camera/
│ └── main.py
├── output/
│ └── cross_camera_result.mp4


---

## ⚙️ Pipeline Overview

1. **Detection**: YOLOv8 model (`best.pt`) is used to detect players in both video feeds.
2. **Embedding Extraction**: Simple RGB average embedding is calculated per player crop.
3. **Re-identification**:
   - Compare each player in `tacticam.mp4` with players in `broadcast.mp4` (first frame).
   - Cosine similarity is used to determine identity match.
4. **Mapping Output**: The mapped identities are printed and optionally visualized.

---

## 📊 Result

| View         | Input              | Output Video              |
|--------------|--------------------|----------------------------|
| Cross-Camera | `broadcast.mp4`, `tacticam.mp4` | `cross_camera_result.mp4`|

---

## 🧠 Limitations
- Uses basic RGB embeddings.
- No spatial calibration between cameras.

---

## 🚀 Future Improvements
- Use deep embeddings from pre-trained ReID models.
- Add camera calibration or homography for better alignment.
- Improve matching with more temporal context.


