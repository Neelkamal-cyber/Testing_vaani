---
title: VAANI Sign Language Recognition
emoji: 🤟
colorFrom: green
colorTo: blue
sdk: streamlit
sdk_version: 1.32.2
app_file: app.py
pinned: false
license: mit
---

# 🤟 VAANI — Real-Time Indian Sign Language → Speech

VAANI converts Indian Sign Language gestures into spoken words in real-time
using only a standard webcam. No special hardware. No app install.

## How it works
1. **MediaPipe Holistic** extracts 258 skeletal keypoints per frame (pose + both hands)
2. A rolling 60-frame window feeds an **Encoder-Decoder LSTM** model
3. Confirmed signs are spoken aloud via **gTTS** (plays directly in the browser)

## Usage
- Click **START** on the camera widget and allow webcam access
- Sign clearly with good lighting
- The detected sign appears on the right panel and is spoken aloud
- Press **🗑️ Clear** to reset the sentence

## Vocabulary (25 ISL signs)
Hello · Thank You · Sorry · Please · Yes · No · Good · Bad ·
Happy · Sad · Pain · Hungry · Thirsty · Water · Food · Help ·
Stop · Go · Come · Mother · Father · Friend · Doctor · School · Hospital

## Performance
| Condition | Accuracy |
|-----------|----------|
| Isolated signs (good lighting) | **95 %** |
| Continuous signing | 85–90 % |
| End-to-end latency | < 20 ms |

## Files required in your Space
```
app.py
requirements.txt
packages.txt
README.md
vaani_endec_deploy.h5      ← upload with Git LFS if larger than 50 MB
label_map.json
```

## Uploading a large model with Git LFS
```bash
git lfs install
git lfs track "*.h5"
git add .gitattributes
git add vaani_endec_deploy.h5
git commit -m "add model via LFS"
git push
```

## Run locally
```bash
pip install -r requirements.txt
streamlit run app.py
```

## Team
Neel Kamal (2203400100036) · Arun Chaudhary (2203400100010) · Chetan Sharma (2203400100015)
Vivekananda College of Technology & Management, Aligarh
B.Tech CSE Final Year Project · 2026
