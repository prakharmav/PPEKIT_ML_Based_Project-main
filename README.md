# 🦺 SafeGuard AI — Industrial PPE Safety Compliance Monitor

A real-time **Personal Protective Equipment (PPE) detection system** powered by YOLOv8, built with a premium dark-themed glassmorphic Streamlit dashboard. Designed for industrial environments to automatically detect safety violations such as missing helmets and safety vests.

---

## 🚀 Features

- **Real-Time Detection** — Live webcam feed inference with bounding box overlays and HUD annotations
- **Image & Video Analysis** — Upload photos or video files for PPE compliance checks
- **Violation Logging** — Automatic incident screenshots saved with timestamps
- **Compliance Dashboard** — Interactive charts, compliance scores, and violation trend analysis
- **Incident Reports** — Export detailed safety reports as **PDF** or **CSV**
- **Premium UI** — Dark glassmorphic design with animated badges, live status indicators, and smooth transitions

---

## 🧠 Tech Stack

| Layer | Technology |
|-------|-----------|
| Detection Model | YOLOv8 (`ultralytics`) |
| Web UI | Streamlit |
| Computer Vision | OpenCV, Pillow |
| Charts & Plots | Plotly |
| Database | SQLite (via custom `database.py`) |
| Report Generation | fpdf2, pandas |

---

## 📁 Project Structure

```
├── app.py                  # Main Streamlit application (multi-page UI)
├── detector.py             # PPE detection engine wrapping YOLOv8
├── database.py             # Incident logging & SQLite database
├── model.py                # Model loader utilities
├── report_generator.py     # PDF & CSV report generation
├── best.pt                 # Trained YOLOv8 weights
├── classes.txt             # Detection class labels
├── incidents/              # Auto-saved violation screenshots
└── README.md
```

---

## ⚙️ Setup & Installation

### 1. Clone the repository

```bash
git clone https://github.com/prakharmav/PPEKIT_ML_Based_Project-main.git
cd PPEKIT_ML_Based_Project-main
```

### 2. Install dependencies

```bash
pip install streamlit opencv-python plotly ultralytics fpdf2 pandas pillow
```

### 3. Run the app

```bash
streamlit run app.py
```

The app will open in your browser at `http://localhost:8501`.

---

## 🔍 Detectable PPE Classes

The model detects the following:

- ✅ **Helmet / Hard Hat** (worn)
- ❌ **No Helmet** (violation)
- ✅ **Safety Vest / Jacket** (worn)
- ❌ **No Vest** (violation)
- 👷 **Person**

---

## 📊 Dashboard Pages

| Page | Description |
|------|-------------|
| **Overview** | Live compliance stats, recent alerts, violation thumbnails |
| **Live Monitor** | Real-time webcam detection stream |
| **Image Analysis** | Upload and analyze single images |
| **Video Analysis** | Upload and process video files |
| **Incident Log** | Browse, filter, and search all recorded violations |
| **Reports** | Generate and download PDF/CSV compliance reports |

---

## 📸 Incident Capture

Whenever a PPE violation is detected, the system automatically:
1. Saves an annotated screenshot to the `incidents/` folder
2. Logs the event (timestamp, violation type, worker count, compliance %) to the database
3. Displays it in the dashboard's incident feed

---

## 🛠️ Configuration

You can adjust detection sensitivity in `detector.py`:

```python
PPEDetector(model_path="best.pt", conf=0.4)  # conf = confidence threshold (0.0 – 1.0)
```

---

## 📄 License

This project is licensed under the MIT License. See [LICENSE](LICENSE) for details.

---

## 👤 Author

**Prakhar Raj**  
GitHub: [@prakharmav](https://github.com/prakharmav)
