# Cross-Modal Audio-Video Summarization - Final Report

Date: 2026-02-13

## 1) Executive Summary
This project implements a cross-modal summarization system that combines audio emotion signals, visual frames, and large language model (LLM) prompting to produce narrative summaries and highlights for videos. A related Whisper-based pipeline provides transcript-driven emotion trajectories, visual analysis plots, and extended reports. The outputs in this workspace include per-video summaries, highlights, and consolidated reports, as well as emotion-based clips and frame extractions.

## 2) Models Used (Detailed)
Core models and methods used across the pipelines:

Audio and emotion modeling
- OpenL3 (optional): audio embeddings for semantic context
- Heuristic audio-emotion mapping: per-second probabilities from audio features
- Audio feature extraction: RMS energy, spectral centroid, zero-crossing rate

Summarization
- LLaMA-family model (HuggingFace `transformers`): cross-modal prompt-to-summary generation

Whisper analysis
- OpenAI Whisper: transcription with timestamps (configurable model size)
- Audio emotion trajectory (Whisper pipeline): anger, happiness, sadness, neutral, fear

Optional text emotion scoring
- Gemini (optional): text emotion scoring for transcript-driven frame targeting
- DistilRoBERTa emotion model (optional): `j-hartmann/emotion-english-distilroberta-base` for transcript emotion scoring

## 3) Methodology Summary (Step by Step)
1) Collect metadata for the target videos (Instruct-V2Xum from HuggingFace).
2) Download videos and extract audio plus 1 FPS frames.
3) Compute audio features and per-second emotion probabilities.
4) Select emotionally salient timestamps and align them to frames.
5) Build prompts that combine audio emotion cues and visual context.
6) Generate summaries with a LLaMA-family model.
7) Produce highlights and consolidated reports from the summaries.
8) (Optional) Run Whisper to transcribe and compute emotion trajectories.
9) (Optional) Generate plots, detailed reports, and emotion-targeted frames/clips.

## 4) System Overview
Two related pipelines run on the same inputs (audio, frames, videos) and produce complementary outputs:

1) Cross-Modal Summarization Pipeline (audio + frames + LLM)
- Extract media
- Compute audio features and emotion probabilities
- Build prompts using audio and visual context
- Generate summaries with a LLaMA-family model
- Produce highlights and final reports

2) Whisper Analysis Pipeline (audio + text + visual trajectories)
- Transcribe audio with Whisper
- Compute audio emotion trajectories with percentages
- Analyze visual brightness and motion trajectories
- Generate plots and comprehensive reports

## 5) Models and Methods
Audio feature extraction
- Per-second features: RMS energy, spectral centroid, zero-crossing rate
- Optional OpenL3 embeddings for richer audio semantics

Heuristic audio-emotion mapping
- Anger: high energy + high spectral centroid
- Happiness: high energy + low zero-crossing
- Sadness: low energy + low spectral centroid
- Neutral: balanced energy
- Fear: high spectral centroid + high zero-crossing

Summarization (cross-modal)
- Prompts fuse audio-emotion peaks with aligned visual frame context
- LLaMA-family model generates 6-8 sentence narrative summaries

Whisper analysis pipeline
- Whisper transcription with timestamps and segment breakdown
- Audio emotion trajectories with percentage statistics (mean, max, min, std dev)
- Visual analysis: brightness trajectory, motion intensity, scene change detection

Optional text emotion analysis
- Gemini text emotion scoring for transcript-driven frame targeting
- DistilRoBERTa emotion classifier for transcript sentiment (optional)

Note: LLaMA models require valid HuggingFace access and a token if the model is gated.

## 6) Cross-Modal Summarization Pipeline Details
Step A: Fetch metadata
- Script: scripts/00_fetch_hf_metadata.py
- Purpose: Pulls dataset metadata (Instruct-V2Xum) from HuggingFace

Step B: Download videos and extract media
- Script: scripts/01_download_and_extract.py
- Outputs:
  - data/videos/ (original videos)
  - data/audio/ (audio WAVs)
  - data/frames/ (1 FPS frame folders per video)

Step C: Compute audio features and emotion probabilities
- scripts/02_audio_features.py
  - Output: audio_features/<video_id>_features.json
  - Fields: per_second features, emotion_probs, highlight scores, optional OpenL3 embeddings
- scripts/03_audio_features.py
  - Output: data/features/<video_id>_audio.json
  - Fields: emotion_probs, OpenL3 embeddings, timestamps, intensity

Step D: Create prompts and summarize
- Script: scripts/03_make_prompts_and_summarize_llama.py
- Outputs:
  - outputs/<video_id>_prompt.txt
  - outputs/<video_id>_summary.txt
  - outputs/<video_id>_summary.json (if created by the script)

Step E: Highlights and reports
- Highlights: scripts/04_highlights.py
- Report: scripts/07_make_report.py
- Outputs:
  - outputs/<video_id>_highlights.json
  - outputs/final_report.csv
  - outputs/final_report.json

## 7) Emotion Clips
Emotion clips are created by scanning per-second emotion probabilities and slicing corresponding frame ranges into video clips.

- All emotions: scripts/create_all_emotion_clips.py
  - Reads: data/features/<video_id>_audio.json
  - Outputs: outputs/emotion_clips/<emotion>/<video_id>_<emotion>_*.mp4
  - Emotions: neutral, happy, sad, angry, surprise

- Happy-only: scripts/create_happiness_clips.py
  - Reads: data/features/<video_id>_audio.json
  - Outputs: outputs/happiness_clips/<video_id>_happy_*.mp4

Clip creation logic:
- Finds segments where a target emotion exceeds a threshold for a minimum duration.
- Copies 1 FPS frames into a temp folder and uses ffmpeg to encode a clip at 24 FPS.

## 8) Whisper Analysis Pipeline Details
Step A: Whisper transcription and emotion trajectory
- Script: scripts/08_whisper_emotion_trajectory.py
- Outputs:
  - outputs/trajectories/<video_id>_analysis.json
  - outputs/trajectories/all_videos_analysis.json

Step B: Visualizations
- Script: scripts/09_visualize_trajectories.py
- Outputs:
  - outputs/visualizations/<video_id>_emotion_trajectory.png
  - outputs/visualizations/<video_id>_visual_analysis.png
  - outputs/visualizations/<video_id>_combined_analysis.png

Step C: Reports
- Script: scripts/10_generate_report.py
- Outputs:
  - outputs/comprehensive_analysis_report.csv
  - outputs/comprehensive_analysis_report.json
  - outputs/detailed_analysis_report.md

Step D: Emotion frames from audio or text
- Audio emotion frames: scripts/11_extract_emotion_frames.py
  - Output: outputs/frames_by_emotion/<video_id>/<emotion>/ + manifest JSON

- Text emotion frames (Gemini): scripts/12_gemini_text_emotion_frames.py
  - Output: outputs/frames_by_text_emotion/<video_id>/<emotion>/ + manifest JSON

## 9) Outputs and Locations (Quick Reference)
- Summaries (text): outputs/<video_id>_summary.txt
- Prompts: outputs/<video_id>_prompt.txt
- Highlights: outputs/<video_id>_highlights.json
- Final report: outputs/final_report.csv, outputs/final_report.json
- Whisper analysis (JSON): outputs/trajectories/<video_id>_analysis.json
- Whisper plots: outputs/visualizations/*.png
- Whisper reports: outputs/comprehensive_analysis_report.csv/.json, outputs/detailed_analysis_report.md
- Emotion clips: outputs/emotion_clips/<emotion>/*.mp4
- Happiness clips: outputs/happiness_clips/*.mp4
- Emotion frames: outputs/frames_by_emotion/<video_id>/<emotion>/
- Text-emotion frames: outputs/frames_by_text_emotion/<video_id>/<emotion>/

## 10) Consolidated Per-Video Results (from outputs/final_report.json)

### fcfQkxwz4Oo
- Top audio frames: 295, 294, 296, 293, 297, 292, 861, 298, 1309, 109
- Summary: The video opens with an early, emotionally resonant moment, setting a foundational tone for the unfolding narrative. A subsequent and extended sequence of interactions then dominates, marked by a rapid succession of intensely emotional audio cues. This critical cluster of moments suggests a central conflict or pivotal decision, unfolding with palpable tension and significant emotional depth. Following this intense period, the narrative progresses to a distinct major event, indicating a crucial turning point or development. Further into the story, another deeply significant occurrence arises, likely a consequence of prior actions or a new challenge. The journey culminates in a powerful final moment, rich with emotional resonance, which provides a profound conclusion to the overarching story. Throughout, the detected emotional intensity from the audio frames serves as the primary guide, highlighting the most impactful events and driving the cause-effect flow of the narrative.

### GM8Yd0j0BFQ
- Top audio frames: 146, 224, 163, 145, 162, 147, 225, 223, 161, 364
- Summary: The narrative began with an undercurrent of suspense, as initial moments hinted at a significant emotional build-up. A pivotal event then dramatically unfolded, immediately sparking a wave of intense reactions across various participants. These heightened emotions, ranging from surprise to profound concern, underscored the critical nature of the situation. Prompted by this turning point, characters made crucial decisions, setting in motion a chain of cause-and-effect that propelled the story forward. The consequences of these actions further intensified the emotional stakes, leading to moments of both distress and determination. The tension ultimately culminated in a powerful emotional release, marking a definitive shift in the story's direction. This journey through varied and impactful emotional beats painted a vivid picture of the challenges faced and the final transformation achieved.

### NAQUNkEtDpo
- Top audio frames: 139, 140, 138, 732, 115, 114, 137, 20, 731, 19
- Summary: Summary could not be generated from indices alone. The prompt requests descriptions or content for the listed moments to produce a coherent narrative.

### pp05vGzvEHc
- Top audio frames: 985, 1066, 986, 1065, 213, 984, 212, 949, 1052, 950
- Summary: The video immediately captures a shift in atmosphere as an unexpected event introduces a demanding challenge, eliciting initial moments of surprise and focused determination. As the stakes rapidly escalated, crucial decisions were made under palpable tension, evidenced by the audio's building intensity and the participants' strained expressions. A critical juncture arrived with a dramatic confrontation against a formidable obstacle, leading to a concentrated period of intense effort and raw emotional outpouring. Through sheer grit and collaborative spirit, the team navigated this immense difficulty, sparking a pivotal turning point in their journey. This struggle culminated in a powerful emotional crescendo, as the final hurdle was overcome with an overwhelming wave of relief and exhilaration. The subsequent scenes erupted in shouts of joy and profound satisfaction, marking a hard-won triumph. Overall, the narrative powerfully conveys a journey from initial shock and mounting pressure to the ultimate, deeply gratifying success, highlighting the significant emotional investment throughout.

### ZmNpeXTj2c4
- Top audio frames: 3697, 5545, 2219, 2220, 5544, 5474, 6054, 5473, 3592, 3700
- Summary: The video immediately immerses viewers in a scene of palpable tension, underscored by an early surge of emotional intensity as a significant challenge is introduced. Key moments of struggle and frustration become evident, particularly during a heated exchange where voices and gestures convey deep-seated concerns, marking a critical phase. A powerful turning point emerges with a crucial decision or revelation, causing a dramatic shift in the atmosphere and eliciting powerful, raw reactions from those involved. This pivotal action directly triggers a cascade of emotions, ranging from shock and apprehension to resolute determination, driving the narrative forward with urgent purpose. As the situation escalates, the heightened emotional states of the participants become central, reflecting the mounting pressure and high stakes involved. Ultimately, the dramatic events culminate in a decisive outcome, accompanied by a final, profound emotional release that reshapes the entire dynamic.

## 11) Entry Points
- Full Whisper analysis pipeline: run_analysis_pipeline.py
- Cross-modal summarization (LLM): scripts/03_make_prompts_and_summarize_llama.py (after media and features)

## 12) Results Summary (Workspace Outputs)
Cross-modal summarization
- Per-video prompts and summaries in outputs/ (e.g., *_prompt.txt, *_summary.txt)
- Highlights JSON per video (outputs/*_highlights.json)
- Consolidated reports: outputs/final_report.csv and outputs/final_report.json

Whisper analysis
- Trajectory JSON per video with transcription, emotion percentages, and visual metrics
- Visualizations per video (emotion trajectory, visual analysis, combined view)
- Consolidated reports: outputs/comprehensive_analysis_report.csv/.json and outputs/detailed_analysis_report.md

Emotion-driven clips and frames
- Emotion clips under outputs/emotion_clips/ and outputs/happiness_clips/
- Emotion frames under outputs/frames_by_emotion/ and outputs/frames_by_text_emotion/

## 13) Notes and Limitations
- Some summaries depend on the availability and quality of audio emotion signals.
- The NAQUNkEtDpo entry requires visual or audio descriptions for the listed moments to generate a valid narrative summary.
- GPU acceleration and large model access may be required for faster processing and gated model usage.
