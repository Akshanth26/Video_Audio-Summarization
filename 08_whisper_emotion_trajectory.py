"""
Comprehensive Video Summarization with Whisper Transcription and Emotion Trajectory
- Uses Whisper for audio transcription
- Analyzes audio emotion trajectory with percentages
- Analyzes video frames with scene detection
- Generates temporal trajectory with confidence percentages
"""

from pathlib import Path
import json
import numpy as np
import librosa
import soundfile as sf
import cv2
from tqdm import tqdm
import warnings
warnings.filterwarnings('ignore')

# Whisper for transcription
try:
    import whisper
    WHISPER_AVAILABLE = True
except ImportError:
    print("Warning: Whisper not installed. Install with: pip install openai-whisper")
    WHISPER_AVAILABLE = False

# Audio emotion detection
try:
    import os
    os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
    from transformers import pipeline
    TRANSFORMERS_AVAILABLE = True
except Exception as e:
    print(f"Warning: Transformers not available: {e}")
    TRANSFORMERS_AVAILABLE = False

# Video directories
VIDEO_DIR = Path("data/videos")
FRAME_DIR = Path("data/frames")
AUDIO_DIR = Path("data/audio")
OUTPUT_DIR = Path("outputs")
TRAJECTORY_DIR = OUTPUT_DIR / "trajectories"

# Create output directories
AUDIO_DIR.mkdir(parents=True, exist_ok=True)
TRAJECTORY_DIR.mkdir(parents=True, exist_ok=True)


class VideoAnalyzer:
    """Comprehensive video analysis with Whisper and emotion tracking"""
    
    def __init__(self, model_size="base"):
        """
        Initialize analyzer with Whisper model
        
        Args:
            model_size: Whisper model size (tiny, base, small, medium, large)
        """
        self.model_size = model_size
        
        if WHISPER_AVAILABLE:
            print(f"Loading Whisper model: {model_size}")
            self.whisper_model = whisper.load_model(model_size)
        else:
            self.whisper_model = None
            
        # Try to load emotion classifier
        self.emotion_classifier = None
        # Disable text emotion classifier to avoid TensorFlow issues
        # if TRANSFORMERS_AVAILABLE:
        #     try:
        #         print("Loading emotion classifier...")
        #         self.emotion_classifier = pipeline(
        #             "text-classification",
        #             model="j-hartmann/emotion-english-distilroberta-base",
        #             top_k=None
        #         )
        #     except Exception as e:
        #         print(f"Could not load emotion classifier: {e}")
    
    def extract_audio(self, video_path, audio_path):
        """Extract audio from video file"""
        import subprocess
        
        cmd = [
            'ffmpeg', '-i', str(video_path),
            '-vn', '-acodec', 'pcm_s16le',
            '-ar', '16000', '-ac', '1',
            '-y', str(audio_path)
        ]
        
        try:
            subprocess.run(cmd, check=True, capture_output=True)
            return True
        except subprocess.CalledProcessError as e:
            print(f"Error extracting audio: {e}")
            return False
    
    def transcribe_with_whisper(self, audio_path):
        """Transcribe audio using Whisper with timestamps"""
        if not self.whisper_model:
            return None
        
        print("Transcribing audio with Whisper...")
        result = self.whisper_model.transcribe(
            str(audio_path),
            verbose=False,
            word_timestamps=True
        )
        
        return result
    
    def analyze_audio_emotions(self, audio_path, window_duration=2.0):
        """
        Analyze audio emotions using acoustic features
        Returns trajectory of emotions with percentages
        """
        print("Analyzing audio emotions...")
        
        # Load audio
        y, sr = librosa.load(audio_path, sr=16000, mono=True)
        duration = len(y) / sr
        
        # Window settings
        window_samples = int(window_duration * sr)
        hop_samples = window_samples // 2
        
        trajectories = {
            'anger': [],
            'happiness': [],
            'sadness': [],
            'neutral': [],
            'fear': [],
            'timestamps': []
        }
        
        # Process in windows
        for start_sample in range(0, len(y) - window_samples, hop_samples):
            end_sample = start_sample + window_samples
            window = y[start_sample:end_sample]
            timestamp = start_sample / sr
            
            # Extract acoustic features
            emotions = self._extract_emotion_features(window, sr)
            
            trajectories['anger'].append(emotions['anger'])
            trajectories['happiness'].append(emotions['happiness'])
            trajectories['sadness'].append(emotions['sadness'])
            trajectories['neutral'].append(emotions['neutral'])
            trajectories['fear'].append(emotions['fear'])
            trajectories['timestamps'].append(timestamp)
        
        # Calculate percentages over time
        trajectories['percentages'] = self._calculate_emotion_percentages(trajectories)
        
        return trajectories
    
    def _extract_emotion_features(self, audio_window, sr):
        """Extract emotion features from audio window"""
        
        # RMS Energy (volume/intensity)
        rms = librosa.feature.rms(y=audio_window)[0].mean()
        
        # Zero Crossing Rate (noisiness)
        zcr = librosa.feature.zero_crossing_rate(audio_window)[0].mean()
        
        # Spectral Centroid (brightness)
        spectral_centroid = librosa.feature.spectral_centroid(
            y=audio_window, sr=sr
        )[0].mean()
        
        # Normalize features
        rms_norm = min(rms * 10, 1.0)  # Normalize to 0-1
        zcr_norm = min(zcr * 2, 1.0)
        sc_norm = min(spectral_centroid / 4000, 1.0)
        
        # Map acoustic features to emotions (heuristic model)
        emotions = {
            'anger': 0.6 * rms_norm + 0.4 * sc_norm,
            'happiness': 0.7 * rms_norm + 0.3 * (1 - zcr_norm),
            'sadness': 0.8 * (1 - rms_norm) + 0.2 * (1 - sc_norm),
            'neutral': 1 - abs(rms_norm - 0.5),
            'fear': 0.5 * sc_norm + 0.5 * zcr_norm
        }
        
        # Normalize to sum to 1 (percentages)
        total = sum(emotions.values())
        emotions = {k: v / total for k, v in emotions.items()}
        
        return emotions
    
    def _calculate_emotion_percentages(self, trajectories):
        """Calculate emotion percentages for entire video"""
        
        percentages = {}
        for emotion in ['anger', 'happiness', 'sadness', 'neutral', 'fear']:
            values = np.array(trajectories[emotion])
            percentages[emotion] = {
                'mean': float(np.mean(values) * 100),
                'max': float(np.max(values) * 100),
                'min': float(np.min(values) * 100),
                'std': float(np.std(values) * 100)
            }
        
        # Overall dominant emotion
        mean_values = {k: v['mean'] for k, v in percentages.items()}
        dominant_emotion = max(mean_values, key=mean_values.get)
        
        return {
            'by_emotion': percentages,
            'dominant_emotion': dominant_emotion,
            'dominant_percentage': mean_values[dominant_emotion]
        }
    
    def analyze_text_emotions(self, text_segments):
        """Analyze emotions in transcribed text"""
        if not self.emotion_classifier or not text_segments:
            return None
        
        print("Analyzing text emotions...")
        
        text_emotions = []
        for segment in tqdm(text_segments, desc="Text emotion analysis"):
            text = segment.get('text', '').strip()
            if not text:
                continue
            
            try:
                emotions = self.emotion_classifier(text)[0]
                emotion_dict = {e['label']: e['score'] for e in emotions}
                
                text_emotions.append({
                    'text': text,
                    'start': segment.get('start'),
                    'end': segment.get('end'),
                    'emotions': emotion_dict,
                    'dominant_emotion': max(emotion_dict, key=emotion_dict.get)
                })
            except Exception as e:
                print(f"Error analyzing text emotion: {e}")
                continue
        
        return text_emotions
    
    def analyze_frames(self, video_path):
        """Analyze video frames for scene changes and visual content"""
        print("Analyzing video frames...")
        
        cap = cv2.VideoCapture(str(video_path))
        if not cap.isOpened():
            print(f"Error opening video: {video_path}")
            return None
        
        fps = cap.get(cv2.CAP_PROP_FPS)
        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        duration = frame_count / fps if fps > 0 else 0
        
        # Sample frames (every 1 second)
        sample_interval = int(fps)
        
        frame_analysis = {
            'fps': fps,
            'total_frames': frame_count,
            'duration': duration,
            'scenes': [],
            'brightness_trajectory': [],
            'motion_trajectory': [],
            'timestamps': []
        }
        
        prev_frame = None
        frame_idx = 0
        
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            
            if frame_idx % sample_interval == 0:
                timestamp = frame_idx / fps
                
                # Convert to grayscale
                gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                
                # Calculate brightness
                brightness = np.mean(gray) / 255.0
                
                # Calculate motion (if previous frame exists)
                motion = 0.0
                if prev_frame is not None:
                    diff = cv2.absdiff(gray, prev_frame)
                    motion = np.mean(diff) / 255.0
                
                frame_analysis['brightness_trajectory'].append(brightness)
                frame_analysis['motion_trajectory'].append(motion)
                frame_analysis['timestamps'].append(timestamp)
                
                # Detect scene changes (high motion)
                if motion > 0.15:
                    frame_analysis['scenes'].append({
                        'timestamp': timestamp,
                        'frame': frame_idx,
                        'motion_intensity': float(motion),
                        'brightness': float(brightness)
                    })
                
                prev_frame = gray.copy()
            
            frame_idx += 1
        
        cap.release()
        
        # Calculate visual statistics
        frame_analysis['visual_stats'] = {
            'avg_brightness': float(np.mean(frame_analysis['brightness_trajectory'])),
            'avg_motion': float(np.mean(frame_analysis['motion_trajectory'])),
            'scene_changes': len(frame_analysis['scenes']),
            'scene_change_percentage': (len(frame_analysis['scenes']) / (duration + 0.001)) * 100
        }
        
        return frame_analysis
    
    def create_comprehensive_summary(self, video_id, transcription, 
                                     audio_emotions, text_emotions, 
                                     frame_analysis):
        """Create comprehensive summary with all trajectories"""
        
        summary = {
            'video_id': video_id,
            'transcription': {
                'text': transcription.get('text', '') if transcription else '',
                'language': transcription.get('language', '') if transcription else '',
                'segments': []
            },
            'audio_emotion_trajectory': audio_emotions,
            'text_emotion_analysis': text_emotions,
            'visual_analysis': frame_analysis,
            'key_insights': {}
        }
        
        # Add transcription segments
        if transcription and 'segments' in transcription:
            for seg in transcription['segments']:
                summary['transcription']['segments'].append({
                    'start': seg['start'],
                    'end': seg['end'],
                    'text': seg['text']
                })
        
        # Generate key insights
        if audio_emotions:
            dominant = audio_emotions['percentages']['dominant_emotion']
            dominant_pct = audio_emotions['percentages']['dominant_percentage']
            
            summary['key_insights']['dominant_emotion'] = dominant
            summary['key_insights']['dominant_emotion_percentage'] = f"{dominant_pct:.1f}%"
        
        if frame_analysis:
            summary['key_insights']['total_scenes'] = frame_analysis['visual_stats']['scene_changes']
            summary['key_insights']['avg_brightness'] = f"{frame_analysis['visual_stats']['avg_brightness']*100:.1f}%"
            summary['key_insights']['avg_motion'] = f"{frame_analysis['visual_stats']['avg_motion']*100:.1f}%"
        
        return summary
    
    def process_video(self, video_path):
        """Process a single video completely"""
        video_id = video_path.stem
        print(f"\n{'='*60}")
        print(f"Processing: {video_id}")
        print(f"{'='*60}")
        
        # Extract audio
        audio_path = AUDIO_DIR / f"{video_id}.wav"
        if not audio_path.exists():
            print("Extracting audio...")
            if not self.extract_audio(video_path, audio_path):
                print("Failed to extract audio")
                return None
        
        # Transcribe with Whisper
        transcription = None
        if WHISPER_AVAILABLE:
            transcription = self.transcribe_with_whisper(audio_path)
        
        # Analyze audio emotions
        audio_emotions = self.analyze_audio_emotions(audio_path)
        
        # Analyze text emotions
        text_emotions = None
        if transcription and 'segments' in transcription:
            text_emotions = self.analyze_text_emotions(transcription['segments'])
        
        # Analyze frames
        frame_analysis = self.analyze_frames(video_path)
        
        # Create comprehensive summary
        summary = self.create_comprehensive_summary(
            video_id, transcription, audio_emotions, 
            text_emotions, frame_analysis
        )
        
        # Save results
        output_path = TRAJECTORY_DIR / f"{video_id}_analysis.json"
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(summary, f, indent=2, ensure_ascii=False)
        
        print(f"\n✓ Analysis saved to: {output_path}")
        
        # Print summary
        self.print_summary(summary)
        
        return summary
    
    def print_summary(self, summary):
        """Print a formatted summary"""
        print("\n" + "="*60)
        print("ANALYSIS SUMMARY")
        print("="*60)
        
        if 'key_insights' in summary:
            insights = summary['key_insights']
            print(f"\n📊 Dominant Emotion: {insights.get('dominant_emotion', 'N/A')} "
                  f"({insights.get('dominant_emotion_percentage', 'N/A')})")
            print(f"🎬 Scene Changes: {insights.get('total_scenes', 'N/A')}")
            print(f"💡 Avg Brightness: {insights.get('avg_brightness', 'N/A')}")
            print(f"🎭 Avg Motion: {insights.get('avg_motion', 'N/A')}")
        
        if 'audio_emotion_trajectory' in summary and summary['audio_emotion_trajectory']:
            print("\n🎵 Audio Emotion Percentages:")
            percentages = summary['audio_emotion_trajectory']['percentages']['by_emotion']
            for emotion, stats in percentages.items():
                print(f"  {emotion.capitalize():12} - Mean: {stats['mean']:5.1f}%, "
                      f"Max: {stats['max']:5.1f}%, Min: {stats['min']:5.1f}%")
        
        if 'transcription' in summary and summary['transcription']['text']:
            text = summary['transcription']['text']
            print(f"\n📝 Transcription ({len(text)} chars):")
            print(f"  {text[:200]}{'...' if len(text) > 200 else ''}")


def main():
    """Main execution function"""
    
    # Check for videos
    video_files = sorted(VIDEO_DIR.glob("*.mp4"))
    if not video_files:
        print(f"No videos found in {VIDEO_DIR}")
        return
    
    print(f"Found {len(video_files)} videos to process")
    
    # Initialize analyzer (use 'base' model, or 'tiny' for faster processing)
    analyzer = VideoAnalyzer(model_size="base")
    
    # Process all videos
    results = []
    for video_path in video_files:
        try:
            result = analyzer.process_video(video_path)
            if result:
                results.append(result)
        except Exception as e:
            print(f"Error processing {video_path.name}: {e}")
            import traceback
            traceback.print_exc()
    
    # Save combined results
    combined_path = TRAJECTORY_DIR / "all_videos_analysis.json"
    with open(combined_path, 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    
    print(f"\n{'='*60}")
    print(f"✓ Processed {len(results)} videos successfully")
    print(f"✓ Combined results saved to: {combined_path}")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
