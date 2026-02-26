# Cross-Modal Video Summarization (Audio-Emotion + Audio-Semantic Alignment)

This repository contains an end-to-end pipeline to reproduce the proposed system that augments video summarization with audio-emotion trajectories and audio-semantic highlight scoring. The pipeline uses Instruct-V2Xum metadata (HuggingFace) and computes per-second audio features (OpenL3 + heuristic emotion trajectory). It then creates temporal prompts and generates summaries with a LLaMA-family model via HuggingFace `transformers`.

Prerequisites
- Python 3.9+
- ffmpeg installed and on PATH
- Recommended: GPU and CUDA for model inference

Install dependencies (example using pip):

```bash
python -m pip install -r requirements.txt
```

Important notes about models
- The script references LLaMA-family models on HuggingFace (e.g., `meta-llama/Llama-2-7b-chat-hf`). You must have proper access to those models and set up an HF token if required.
- For large models, set device to `cuda` and ensure you have enough GPU memory.

Quick run (small subset)

1. Fetch metadata (subset of dataset):

```bash
python scripts/00_fetch_hf_metadata.py --out metadata.csv --max 50
```

2. Download videos and extract media (requires `yt-dlp` and `ffmpeg`):

```bash
python scripts/01_download_and_extract.py --metadata metadata.csv --out_dir . --start 0 --end 10
```

3. Compute audio features and highlights:

```bash
python scripts/02_audio_features.py --audio_dir audio --out_dir audio_features --topk 5
```

4. Generate prompts and summarize using a LLaMA model (example):

```bash
# set a model name you have access to
export LLAMA_MODEL=meta-llama/Llama-2-7b-chat-hf
python scripts/03_make_prompts_and_summarize_llama.py --features_dir audio_features --frames_dir frames --out_dir outputs --model $LLAMA_MODEL
```

Outputs
- `audio_features/<youtube_id>_features.json` — per-second features, emotion probs, highlight scores
- `outputs/<youtube_id>_summary.txt` — generated 6–8 sentence summary
- `outputs/<youtube_id>_summary.json` — metadata + summary

Next steps and improvements
- Replace the heuristic emotion mapping with a pretrained CREMA-D or other emotion classifier.
- Add YAMNet / VGGish for richer audio-semantic features.
- Fuse visual embeddings (CLIP) and openl3 embeddings into learned adapter layers to condition LLM embeddings (requires additional modeling code).
- Add evaluation scripts (ROUGE, CLIP-Score) and human evaluation forms.

If you want, I can now run a small end-to-end demo on 1–2 videos from your metadata (will download videos); should I proceed?