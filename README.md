# Cross-Modal Video Summarization (Audio-Emotion + Audio-Semantic Alignment)

# 🎬 Audio-Video Emotion Summarization

An end-to-end multimedia analysis pipeline that extracts emotional insights from videos by combining **audio, speech, and visual signals**, and generates structured summaries and reports.

> 🎓 Developed as a B.Tech 3rd Year Course Project

---

## 🚀 Overview

This project processes an input video and performs:

- 🎙️ **Speech Transcription** with timestamps  
- 🎧 **Audio Emotion Analysis** (temporal emotion trajectory)  
- 🎥 **Visual Feature Extraction** (brightness, motion, scene changes)  
- 📊 **Multi-format Output Generation** (reports, plots, structured data)

The system integrates multiple modalities to produce **emotion-aware summaries** and **analytical reports**.

---

## ✨ Key Features

- 🔁 **End-to-End Pipeline** (Video → Insights → Report)
- 📈 **Emotion Trajectory Tracking**  
  - Anger, Happiness, Sadness, Fear, Neutral
- 🧠 **Speech-to-Text using Whisper**
- 👁️ **Frame-level Visual Analysis**
- 📦 **Multiple Output Formats**
  - JSON, CSV, Markdown, PNG
- ⚡ **Modular & Scalable Design**

---

## 🛠️ Tech Stack

| Category            | Tools Used |
|--------------------|-----------|
| Programming        | Python 3.9+ |
| Speech Processing  | OpenAI Whisper |
| Audio Analysis     | Librosa |
| Computer Vision    | OpenCV |
| Data Processing    | Pandas |
| Visualization      | Matplotlib |
| Media Handling     | FFmpeg |

---

## 📁 Project Structure


├── scripts/
│ ├── 08_whisper_emotion_trajectory.py # Audio + transcription analysis
│ ├── 09_visualize_trajectories.py # Plot generation
│ └── 10_generate_report.py # Final report creation
│
├── run_analysis_pipeline.py # One-click pipeline execution
├── outputs/ # Generated reports & visualizations
├── data/ # Input videos & intermediate files
├── docs/
│ ├── PROCESS.md
│ ├── RESULTS.md
│
└── README.md


---

## ⚙️ Installation

### 1. Clone the repository
```bash
git clone https://github.com/your-username/your-repo-name.git
cd your-repo-name
2. Install dependencies
pip install -r requirements.txt
3. Install FFmpeg

Windows (Chocolatey):

choco install ffmpeg
▶️ Usage
🔹 Run Complete Pipeline (Recommended)
python run_analysis_pipeline.py
🔹 Run Step-by-Step
python scripts/08_whisper_emotion_trajectory.py
python scripts/09_visualize_trajectories.py
python scripts/10_generate_report.py
📊 Outputs

All results are saved in the outputs/ directory:

📄 Structured Reports (Markdown)
📈 Emotion Trajectory Graphs (PNG)
📁 Data Files (JSON, CSV)

📌 Example outputs: See docs/RESULTS.md
