# Project Process

This document explains the end-to-end workflow used in the project, written in a simple B.Tech 3rd year style.

## 1. Problem Statement

We want to summarize a video by combining audio emotion signals, speech transcript, and visual cues. The final output is a short report and plots that show emotion trajectory over time.

## 2. Dataset and Inputs

- Input videos are stored in data/videos/
- FFmpeg is used to extract audio and frames
- Whisper creates the transcript with timestamps

## 3. Pipeline Steps

1. Audio extraction
   - Convert video to WAV using FFmpeg

2. Whisper transcription
   - Generate text with timestamps

3. Audio emotion trajectory
   - Use audio features to estimate emotion percentages
   - Produce per-time-step emotion values

4. Visual analysis
   - Compute brightness, motion intensity, and scene changes

5. Visualization and reporting
   - Generate plots (emotion curves, bar charts)
   - Produce JSON, CSV, and Markdown reports

## 4. How to Run

- One command:

```bash
python run_analysis_pipeline.py
```

- Step by step:

```bash
python scripts/08_whisper_emotion_trajectory.py
python scripts/09_visualize_trajectories.py
python scripts/10_generate_report.py
```

## 5. Output Files (Summary)

- outputs/trajectories/ - per-video JSON analysis
- outputs/visualizations/ - PNG plots
- outputs/comprehensive_analysis_report.csv
- outputs/comprehensive_analysis_report.json
- outputs/detailed_analysis_report.md

## 6. Notes

- The emotion estimation is heuristic, not a trained classifier
- Processing time depends on video length and Whisper model size
