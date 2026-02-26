# Video Summarization with Whisper & Emotion Trajectory Analysis

Comprehensive video analysis system using OpenAI's Whisper for transcription, combined with audio emotion analysis and visual frame trajectory tracking with detailed percentages.

## Features

### 🎙️ Whisper Transcription
- High-quality audio transcription using OpenAI Whisper
- Word-level timestamps
- Multi-language support
- Segment-based analysis

### 🎭 Audio Emotion Analysis
- Real-time emotion trajectory tracking
- 5 emotion categories: Anger, Happiness, Sadness, Neutral, Fear
- Percentage-based confidence scores
- Temporal dynamics visualization
- Statistical analysis (mean, max, min, std dev)

### 🎬 Frame Analysis
- Brightness trajectory tracking
- Motion intensity analysis
- Automatic scene change detection
- Frame-by-frame visual statistics

### 📊 Comprehensive Reporting
- JSON analysis files with full trajectories
- CSV summary reports
- Detailed markdown reports
- Beautiful visualizations (matplotlib)

## Installation

### 1. Install Dependencies

```bash
pip install -r requirements.txt
```

### 2. Install FFmpeg (Required)

**Windows:**
```bash
# Using chocolatey
choco install ffmpeg

# Or download from: https://ffmpeg.org/download.html
```

**Mac:**
```bash
brew install ffmpeg
```

**Linux:**
```bash
sudo apt-get install ffmpeg
```

## Usage

### Quick Start - Run Complete Pipeline

```bash
python run_analysis_pipeline.py
```

This will:
1. Extract and transcribe audio with Whisper
2. Analyze emotion trajectories
3. Analyze video frames
4. Generate visualizations
5. Create comprehensive reports

### Individual Scripts

#### 1. Whisper Transcription & Emotion Analysis

```bash
python scripts/08_whisper_emotion_trajectory.py
```

**Outputs:**
- `outputs/trajectories/{video_id}_analysis.json` - Complete analysis with:
  - Full transcription with timestamps
  - Emotion trajectories with percentages
  - Frame analysis with scene changes
  - Statistical summaries

**Analysis includes:**
- Audio emotion percentages (mean, max, min, std)
- Dominant emotion identification
- Visual statistics (brightness, motion, scene changes)

#### 2. Visualize Trajectories

```bash
python scripts/09_visualize_trajectories.py
```

**Generates:**
- `outputs/visualizations/{video_id}_emotion_trajectory.png`
  - Stacked area chart of emotions over time
  - Individual emotion line plots
  - Pie chart of emotion distribution
  - Bar chart of emotion statistics
  - Dominant emotion timeline

- `outputs/visualizations/{video_id}_visual_analysis.png`
  - Brightness trajectory
  - Motion intensity over time
  - Scene change markers

- `outputs/visualizations/{video_id}_combined_analysis.png`
  - All-in-one comprehensive visualization
  - Key insights panel
  - Transcription preview

#### 3. Generate Reports

```bash
python scripts/10_generate_report.py
```

**Creates:**
- `outputs/comprehensive_analysis_report.csv` - Tabular summary
- `outputs/comprehensive_analysis_report.json` - Structured data
- `outputs/detailed_analysis_report.md` - Full markdown report

## Output Structure

```
outputs/
├── trajectories/
│   ├── {video_id}_analysis.json       # Individual analysis
│   └── all_videos_analysis.json       # Combined results
├── visualizations/
│   ├── {video_id}_emotion_trajectory.png
│   ├── {video_id}_visual_analysis.png
│   └── {video_id}_combined_analysis.png
├── comprehensive_analysis_report.csv
├── comprehensive_analysis_report.json
└── detailed_analysis_report.md
```

## Analysis JSON Structure

```json
{
  "video_id": "example_video",
  "transcription": {
    "text": "Full transcription...",
    "language": "en",
    "segments": [
      {
        "start": 0.0,
        "end": 2.5,
        "text": "Segment text..."
      }
    ]
  },
  "audio_emotion_trajectory": {
    "anger": [0.15, 0.18, ...],
    "happiness": [0.45, 0.42, ...],
    "sadness": [0.10, 0.12, ...],
    "neutral": [0.20, 0.18, ...],
    "fear": [0.10, 0.10, ...],
    "timestamps": [0.0, 2.0, 4.0, ...],
    "percentages": {
      "by_emotion": {
        "happiness": {
          "mean": 42.5,
          "max": 68.3,
          "min": 15.2,
          "std": 12.4
        },
        ...
      },
      "dominant_emotion": "happiness",
      "dominant_percentage": 42.5
    }
  },
  "visual_analysis": {
    "duration": 120.5,
    "fps": 30.0,
    "brightness_trajectory": [0.65, 0.68, ...],
    "motion_trajectory": [0.15, 0.22, ...],
    "scenes": [
      {
        "timestamp": 15.2,
        "motion_intensity": 0.28,
        "brightness": 0.72
      }
    ],
    "visual_stats": {
      "avg_brightness": 0.67,
      "avg_motion": 0.18,
      "scene_changes": 12
    }
  },
  "key_insights": {
    "dominant_emotion": "happiness",
    "dominant_emotion_percentage": "42.5%",
    "total_scenes": 12,
    "avg_brightness": "67.0%",
    "avg_motion": "18.0%"
  }
}
```

## Configuration

### Whisper Model Size

In `scripts/08_whisper_emotion_trajectory.py`, line 422:

```python
# Options: tiny, base, small, medium, large
analyzer = VideoAnalyzer(model_size="base")
```

**Model Comparison:**
| Model  | Parameters | Speed | Accuracy | VRAM   |
|--------|------------|-------|----------|--------|
| tiny   | 39M        | ~32x  | Good     | ~1GB   |
| base   | 74M        | ~16x  | Better   | ~1GB   |
| small  | 244M       | ~6x   | Great    | ~2GB   |
| medium | 769M       | ~2x   | Excellent| ~5GB   |
| large  | 1550M      | 1x    | Best     | ~10GB  |

### Emotion Analysis Window

Adjust temporal resolution in `scripts/08_whisper_emotion_trajectory.py`, line 103:

```python
def analyze_audio_emotions(self, audio_path, window_duration=2.0):
    # window_duration: seconds per emotion sample
    # Lower = more granular, Higher = smoother trajectory
```

## Emotion Detection Method

### Audio-Based Emotions
Uses acoustic features to estimate emotions:

- **Anger**: High energy + high spectral centroid (loud, bright)
- **Happiness**: High energy + low zero-crossing (loud, smooth)
- **Sadness**: Low energy + low spectral centroid (quiet, dull)
- **Neutral**: Balanced energy levels
- **Fear**: High spectral centroid + high zero-crossing (bright, noisy)

### Text-Based Emotions (Optional)
When enabled, uses transformer model `j-hartmann/emotion-english-distilroberta-base` for sentiment analysis of transcribed text.

## Example Workflow

```bash
# 1. Place videos in data/videos/
cp my_video.mp4 data/videos/

# 2. Run complete pipeline
python run_analysis_pipeline.py

# 3. View results
# - Check outputs/visualizations/ for plots
# - Read outputs/detailed_analysis_report.md
# - Open outputs/comprehensive_analysis_report.csv in Excel
```

## Visualizations

### Emotion Trajectory Plot
- **Stacked Area Chart**: Shows emotion distribution over time (0-100%)
- **Individual Lines**: Track each emotion independently
- **Pie Chart**: Overall emotion distribution
- **Statistics Bar Chart**: Mean, Max, Min values
- **Timeline**: Color-coded dominant emotion at each moment

### Visual Analysis Plot
- **Brightness Trajectory**: Video lighting levels over time
- **Motion Intensity**: Movement detection across frames
- **Scene Changes**: Automatic detection of cuts/transitions

### Combined Analysis
Single comprehensive view with:
- All emotion trajectories
- Visual metrics
- Key insights panel
- Transcription preview

## Requirements

- Python 3.8+
- FFmpeg
- ~2-10GB VRAM (depending on Whisper model size)
- Disk space for video files and outputs

## Troubleshooting

### Issue: "No module named 'whisper'"
```bash
pip install openai-whisper
```

### Issue: "FFmpeg not found"
Install FFmpeg (see Installation section)

### Issue: Out of memory
Use smaller Whisper model:
```python
analyzer = VideoAnalyzer(model_size="tiny")  # or "base"
```

### Issue: Slow processing
- Use "tiny" or "base" Whisper model
- Process videos in batches
- Reduce frame sampling rate in frame analysis

## Performance Tips

1. **Faster Transcription**: Use "tiny" or "base" models
2. **GPU Acceleration**: CUDA-enabled PyTorch will automatically use GPU
3. **Batch Processing**: Process multiple videos in sequence
4. **Skip Visualization**: Comment out visualization step if not needed

## Advanced Usage

### Process Single Video

```python
from pathlib import Path
from scripts.08_whisper_emotion_trajectory import VideoAnalyzer

analyzer = VideoAnalyzer(model_size="base")
video_path = Path("data/videos/my_video.mp4")
result = analyzer.process_video(video_path)
```

### Custom Emotion Weights

Modify emotion calculation in `_extract_emotion_features()`:

```python
emotions = {
    'anger': 0.7 * rms_norm + 0.3 * sc_norm,  # More weight on volume
    'happiness': 0.8 * rms_norm + 0.2 * (1 - zcr_norm),
    # ... adjust weights as needed
}
```

## Citation

If you use this code in your research, please cite:

```bibtex
@software{whisper_emotion_trajectory,
  title={Video Summarization with Whisper and Emotion Trajectory Analysis},
  author={Your Name},
  year={2025},
  url={https://github.com/yourusername/audio_video_summarization}
}
```

## License

MIT License - Feel free to use and modify

## Acknowledgments

- OpenAI Whisper for transcription
- Librosa for audio analysis
- Transformers for emotion classification
- OpenCV for video processing

## Contact

For questions or issues, please open an issue on GitHub.
