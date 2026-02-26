# Quick Start Guide - Whisper Video Analysis

## 🚀 Setup (5 minutes)

### 1. Install Dependencies
```bash
pip install openai-whisper opencv-python librosa soundfile transformers matplotlib pandas numpy tqdm
```

### 2. Install FFmpeg
**Windows (chocolatey):**
```bash
choco install ffmpeg
```

**Or download**: https://ffmpeg.org/download.html

## 🎬 Run Analysis

### Option 1: Complete Pipeline (Recommended)
```bash
python run_analysis_pipeline.py
```

This runs everything automatically:
✅ Whisper transcription  
✅ Emotion trajectory analysis  
✅ Frame analysis  
✅ Visualizations  
✅ Reports  

### Option 2: Step by Step

**Step 1: Analyze with Whisper**
```bash
python scripts/08_whisper_emotion_trajectory.py
```

**Step 2: Create Visualizations**
```bash
python scripts/09_visualize_trajectories.py
```

**Step 3: Generate Reports**
```bash
python scripts/10_generate_report.py
```

## 📂 Where to Find Results

After running, check these folders:

```
outputs/
├── trajectories/              # JSON analysis files
│   └── {video}_analysis.json
├── visualizations/            # PNG plots
│   ├── {video}_emotion_trajectory.png
│   ├── {video}_visual_analysis.png
│   └── {video}_combined_analysis.png
├── comprehensive_analysis_report.csv    # Excel-friendly
├── comprehensive_analysis_report.json   # Structured data
└── detailed_analysis_report.md         # Human-readable

## Extract Emotion-Based Frames

After running the analysis pipeline (or `scripts/08_whisper_emotion_trajectory.py`), you can extract the most emotional moments as frames:

```
python scripts/11_extract_emotion_frames.py <VIDEO_ID> <EMOTION> --top-k 12 --min-gap-sec 4
```

- `VIDEO_ID`: e.g. `ZmNpeXTj2c4`
- `EMOTION`: one of `happiness`, `anger`, `sadness`, `neutral`, `fear`
- Output frames will be saved under `outputs/frames_by_emotion/<VIDEO_ID>/<EMOTION>/` with a JSON manifest alongside.

Notes:
- The extractor maps emotion timestamps to the nearest available frames in `data/frames/<VIDEO_ID>/`.
- If your frames are downsampled (e.g. only a few hundred), multiple moments may map to the same frame index.

## Gemini Text Emotion → Frames

Use Gemini to score transcript segments for a target emotion and extract aligned frames:

```
python scripts/12_gemini_text_emotion_frames.py <VIDEO_ID> <EMOTION> --top-k 12 --min-gap-sec 5 --dry-run
```

- Remove `--dry-run` once `GOOGLE_API_KEY` is set in your environment and `google-generativeai` is installed.
- Frames output to `outputs/frames_by_text_emotion/<VIDEO_ID>/<EMOTION>/` with a manifest JSON.
- Supported emotions: happiness, anger, sadness, neutral, fear.

Setup:
```
pip install -r requirements.txt
set GOOGLE_API_KEY=your_key_here   # Windows
# or: export GOOGLE_API_KEY=your_key_here  # macOS/Linux
```
```

## 📊 What You Get

### For Each Video:

1. **Emotion Trajectory with Percentages**
   - Anger, Happiness, Sadness, Neutral, Fear
   - Mean, Max, Min, Std Dev for each
   - Dominant emotion identification

2. **Whisper Transcription**
   - Full text with timestamps
   - Segment breakdown
   - Language detection

3. **Visual Analysis**
   - Brightness trajectory (%)
   - Motion intensity (%)
   - Scene change detection
   - Frame-by-frame statistics

4. **Beautiful Visualizations**
   - Stacked emotion charts
   - Individual emotion lines
   - Visual metric plots
   - Combined overview

## ⚡ Quick Examples

### Process Your Own Video
```bash
# 1. Copy video to data/videos/
cp my_video.mp4 data/videos/

# 2. Run pipeline
python run_analysis_pipeline.py

# 3. Check outputs/visualizations/
```

### Change Whisper Model Speed
Edit `scripts/08_whisper_emotion_trajectory.py` line 422:

```python
# Faster (less accurate)
analyzer = VideoAnalyzer(model_size="tiny")

# Balanced (recommended)
analyzer = VideoAnalyzer(model_size="base")

# Better quality (slower)
analyzer = VideoAnalyzer(model_size="small")
```

## 🎯 Key Features

✨ **Emotion Percentages**: See exactly what % of time each emotion dominates  
✨ **Temporal Trajectories**: Watch emotions change over time  
✨ **Scene Detection**: Automatic identification of video cuts  
✨ **Multi-format Output**: JSON, CSV, Markdown, PNG visualizations  

## 🔍 Example Output

```
ANALYSIS SUMMARY
=================================================================
📊 Dominant Emotion: happiness (42.5%)
🎬 Scene Changes: 12
💡 Avg Brightness: 67.0%
🎭 Avg Motion: 18.0%

🎵 Audio Emotion Percentages:
  Anger        - Mean:  15.2%, Max:  28.5%, Min:   5.1%
  Happiness    - Mean:  42.5%, Max:  68.3%, Min:  18.2%
  Sadness      - Mean:  10.8%, Max:  22.1%, Min:   2.3%
  Neutral      - Mean:  21.5%, Max:  35.0%, Min:  12.0%
  Fear         - Mean:  10.0%, Max:  18.5%, Min:   4.2%

📝 Transcription (1542 chars):
  Hello everyone, welcome to today's video where we'll be...
```

## ⚙️ Performance Tips

**Faster Processing:**
- Use `model_size="tiny"` (32x faster)
- Process one video at a time
- Skip visualization if not needed

**Better Quality:**
- Use `model_size="medium"` or "large"
- Requires more VRAM (5-10GB)

**Balanced (Recommended):**
- Use `model_size="base"`
- Good quality, reasonable speed
- Works on most systems

## 🐛 Common Issues

**"No videos found"**
→ Place .mp4 files in `data/videos/` folder

**"FFmpeg not found"**
→ Install FFmpeg (see setup)

**"Out of memory"**
→ Use smaller Whisper model: `model_size="tiny"`

**Slow processing**
→ Normal! Whisper transcription takes time
→ Use "tiny" model for testing

## 📧 Need Help?

Check the full documentation in `WHISPER_ANALYSIS_README.md`
