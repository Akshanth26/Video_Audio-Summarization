"""
Generate comprehensive report with all analysis results
Including trajectories, percentages, and insights
"""

from pathlib import Path
import json
import pandas as pd
from datetime import datetime

TRAJECTORY_DIR = Path("outputs/trajectories")
OUTPUT_DIR = Path("outputs")


def create_summary_report():
    """Create comprehensive summary report"""
    
    # Load all analysis files
    analysis_files = sorted(TRAJECTORY_DIR.glob("*_analysis.json"))
    
    if not analysis_files:
        print(f"No analysis files found in {TRAJECTORY_DIR}")
        return
    
    all_data = []
    
    for analysis_file in analysis_files:
        try:
            with open(analysis_file, 'r', encoding='utf-8') as f:
                data = json.load(f)
            if isinstance(data, list):
                all_data.extend([item for item in data if isinstance(item, dict)])
            elif isinstance(data, dict):
                all_data.append(data)
            else:
                print(f"Skipping unexpected data type in {analysis_file}")
        except Exception as e:
            print(f"Error loading {analysis_file}: {e}")
    
    # Create summary table
    summary_rows = []
    
    for data in all_data:
        if not isinstance(data, dict):
            continue
        video_id = data.get('video_id', 'Unknown')
        
        row = {
            'video_id': video_id,
            'dominant_emotion': '',
            'dominant_emotion_percentage': 0.0,
            'anger_mean': 0.0,
            'happiness_mean': 0.0,
            'sadness_mean': 0.0,
            'neutral_mean': 0.0,
            'fear_mean': 0.0,
            'scene_changes': 0,
            'avg_brightness': 0.0,
            'avg_motion': 0.0,
            'word_count': 0,
            'language': '',
            'duration': 0.0
        }
        
        # Extract emotion data
        if 'audio_emotion_trajectory' in data:
            traj = data['audio_emotion_trajectory']
            if 'percentages' in traj:
                pct = traj['percentages']
                row['dominant_emotion'] = pct.get('dominant_emotion', '')
                row['dominant_emotion_percentage'] = pct.get('dominant_percentage', 0.0)
                
                if 'by_emotion' in pct:
                    for emotion in ['anger', 'happiness', 'sadness', 'neutral', 'fear']:
                        if emotion in pct['by_emotion']:
                            row[f'{emotion}_mean'] = pct['by_emotion'][emotion]['mean']
        
        # Extract visual data
        if 'visual_analysis' in data:
            vis = data['visual_analysis']
            if 'visual_stats' in vis:
                stats = vis['visual_stats']
                row['scene_changes'] = stats.get('scene_changes', 0)
                row['avg_brightness'] = stats.get('avg_brightness', 0.0) * 100
                row['avg_motion'] = stats.get('avg_motion', 0.0) * 100
            
            row['duration'] = vis.get('duration', 0.0)
        
        # Extract transcription data
        if 'transcription' in data:
            trans = data['transcription']
            text = trans.get('text', '')
            row['word_count'] = len(text.split()) if text else 0
            row['language'] = trans.get('language', '')
        
        summary_rows.append(row)
    
    # Create DataFrame
    df = pd.DataFrame(summary_rows)
    
    # Save as CSV
    csv_path = OUTPUT_DIR / "comprehensive_analysis_report.csv"
    df.to_csv(csv_path, index=False)
    print(f"✓ Summary CSV saved: {csv_path}")
    
    # Save as JSON
    json_path = OUTPUT_DIR / "comprehensive_analysis_report.json"
    with open(json_path, 'w', encoding='utf-8') as f:
        json.dump(summary_rows, f, indent=2)
    print(f"✓ Summary JSON saved: {json_path}")
    
    # Create detailed text report
    create_detailed_report(all_data)
    
    # Print summary statistics
    print_summary_statistics(df)
    
    return df


def create_detailed_report(all_data):
    """Create detailed markdown report"""
    
    report_path = OUTPUT_DIR / "detailed_analysis_report.md"
    
    with open(report_path, 'w', encoding='utf-8') as f:
        f.write("# Comprehensive Video Analysis Report\n\n")
        f.write(f"**Generated:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
        f.write(f"**Total Videos Analyzed:** {len(all_data)}\n\n")
        f.write("---\n\n")
        
        for data in all_data:
            if not isinstance(data, dict):
                continue
            video_id = data.get('video_id', 'Unknown')
            
            f.write(f"## Video: {video_id}\n\n")
            
            # Key Insights
            if 'key_insights' in data:
                f.write("### 📊 Key Insights\n\n")
                insights = data['key_insights']
                
                f.write(f"- **Dominant Emotion:** {insights.get('dominant_emotion', 'N/A').upper()} ")
                f.write(f"({insights.get('dominant_emotion_percentage', 'N/A')})\n")
                f.write(f"- **Scene Changes:** {insights.get('total_scenes', 'N/A')}\n")
                f.write(f"- **Average Brightness:** {insights.get('avg_brightness', 'N/A')}\n")
                f.write(f"- **Average Motion:** {insights.get('avg_motion', 'N/A')}\n\n")
            
            # Audio Emotion Analysis
            if 'audio_emotion_trajectory' in data:
                traj = data['audio_emotion_trajectory']
                
                f.write("### 🎵 Audio Emotion Analysis\n\n")
                
                if 'percentages' in traj and 'by_emotion' in traj['percentages']:
                    f.write("| Emotion | Mean % | Max % | Min % | Std Dev |\n")
                    f.write("|---------|--------|-------|-------|----------|\n")
                    
                    percentages = traj['percentages']['by_emotion']
                    for emotion in ['anger', 'happiness', 'sadness', 'neutral', 'fear']:
                        if emotion in percentages:
                            stats = percentages[emotion]
                            f.write(f"| {emotion.capitalize():10} | ")
                            f.write(f"{stats['mean']:6.2f} | ")
                            f.write(f"{stats['max']:5.2f} | ")
                            f.write(f"{stats['min']:5.2f} | ")
                            f.write(f"{stats['std']:8.2f} |\n")
                    
                    f.write("\n")
            
            # Visual Analysis
            if 'visual_analysis' in data:
                vis = data['visual_analysis']
                
                f.write("### 🎬 Visual Analysis\n\n")
                f.write(f"- **Duration:** {vis.get('duration', 0):.2f} seconds\n")
                f.write(f"- **FPS:** {vis.get('fps', 0):.2f}\n")
                f.write(f"- **Total Frames:** {vis.get('total_frames', 0):,}\n")
                
                if 'visual_stats' in vis:
                    stats = vis['visual_stats']
                    f.write(f"- **Average Brightness:** {stats.get('avg_brightness', 0)*100:.2f}%\n")
                    f.write(f"- **Average Motion:** {stats.get('avg_motion', 0)*100:.2f}%\n")
                    f.write(f"- **Scene Changes:** {stats.get('scene_changes', 0)}\n\n")
                
                # Scene change details
                if 'scenes' in vis and vis['scenes']:
                    f.write("#### Scene Changes\n\n")
                    f.write("| Time (s) | Frame | Motion Intensity | Brightness |\n")
                    f.write("|----------|-------|------------------|------------|\n")
                    
                    for scene in vis['scenes'][:10]:  # Show first 10
                        f.write(f"| {scene['timestamp']:8.2f} | ")
                        f.write(f"{scene['frame']:5d} | ")
                        f.write(f"{scene['motion_intensity']*100:16.2f}% | ")
                        f.write(f"{scene['brightness']*100:10.2f}% |\n")
                    
                    if len(vis['scenes']) > 10:
                        f.write(f"\n*... and {len(vis['scenes'])-10} more scene changes*\n")
                    
                    f.write("\n")
            
            # Transcription
            if 'transcription' in data:
                trans = data['transcription']
                text = trans.get('text', '')
                
                f.write("### 📝 Transcription\n\n")
                f.write(f"- **Language:** {trans.get('language', 'N/A').upper()}\n")
                f.write(f"- **Word Count:** {len(text.split()) if text else 0}\n")
                f.write(f"- **Character Count:** {len(text)}\n\n")
                
                if text:
                    # Show preview
                    preview = text[:500]
                    f.write("**Transcript Preview:**\n\n")
                    f.write(f"> {preview}{'...' if len(text) > 500 else ''}\n\n")
                
                # Show segments with timestamps
                if 'segments' in trans and trans['segments']:
                    f.write("#### Transcript Segments (First 5)\n\n")
                    f.write("| Start | End | Text |\n")
                    f.write("|-------|-----|------|\n")
                    
                    for seg in trans['segments'][:5]:
                        start = seg.get('start', 0)
                        end = seg.get('end', 0)
                        text = seg.get('text', '').strip()
                        text_preview = text[:50] + '...' if len(text) > 50 else text
                        f.write(f"| {start:5.1f}s | {end:5.1f}s | {text_preview} |\n")
                    
                    if len(trans['segments']) > 5:
                        f.write(f"\n*... and {len(trans['segments'])-5} more segments*\n")
                    
                    f.write("\n")
            
            # Text emotions (if available)
            if 'text_emotion_analysis' in data and data['text_emotion_analysis']:
                f.write("### 💭 Text Emotion Analysis\n\n")
                
                text_emotions = data['text_emotion_analysis'][:5]  # Show first 5
                
                for te in text_emotions:
                    f.write(f"**[{te.get('start', 0):.1f}s - {te.get('end', 0):.1f}s]** ")
                    f.write(f"Dominant: {te.get('dominant_emotion', 'N/A')}\n\n")
                    f.write(f"> {te.get('text', '')}\n\n")
                    
                    if 'emotions' in te:
                        emotions = te['emotions']
                        f.write("Emotion scores: ")
                        f.write(", ".join([f"{k}: {v*100:.1f}%" for k, v in emotions.items()]))
                        f.write("\n\n")
                
                if len(data['text_emotion_analysis']) > 5:
                    f.write(f"*... and {len(data['text_emotion_analysis'])-5} more text segments*\n\n")
            
            f.write("---\n\n")
    
    print(f"✓ Detailed report saved: {report_path}")


def print_summary_statistics(df):
    """Print summary statistics to console"""
    
    print("\n" + "="*70)
    print("SUMMARY STATISTICS")
    print("="*70)
    
    print(f"\nTotal Videos: {len(df)}")
    
    # Emotion statistics
    print("\n--- Emotion Distribution ---")
    for emotion in ['anger', 'happiness', 'sadness', 'neutral', 'fear']:
        col = f'{emotion}_mean'
        if col in df.columns:
            mean = df[col].mean()
            print(f"{emotion.capitalize():12} - Average: {mean:6.2f}%")
    
    # Dominant emotions
    print("\n--- Dominant Emotions ---")
    if 'dominant_emotion' in df.columns:
        emotion_counts = df['dominant_emotion'].value_counts()
        for emotion, count in emotion_counts.items():
            percentage = (count / len(df)) * 100
            print(f"{emotion.capitalize():12} - {count} videos ({percentage:.1f}%)")
    
    # Visual statistics
    print("\n--- Visual Statistics ---")
    if 'avg_brightness' in df.columns:
        print(f"Average Brightness: {df['avg_brightness'].mean():.2f}%")
    if 'avg_motion' in df.columns:
        print(f"Average Motion:     {df['avg_motion'].mean():.2f}%")
    if 'scene_changes' in df.columns:
        print(f"Total Scene Changes: {df['scene_changes'].sum()}")
        print(f"Avg Scene Changes:   {df['scene_changes'].mean():.2f} per video")
    
    # Transcription statistics
    print("\n--- Transcription Statistics ---")
    if 'word_count' in df.columns:
        total_words = df['word_count'].sum()
        avg_words = df['word_count'].mean()
        print(f"Total Words: {total_words:,}")
        print(f"Average Words per Video: {avg_words:.0f}")
    
    if 'duration' in df.columns:
        total_duration = df['duration'].sum()
        avg_duration = df['duration'].mean()
        print(f"Total Duration: {total_duration:.1f}s ({total_duration/60:.1f} minutes)")
        print(f"Average Duration: {avg_duration:.1f}s")
    
    print("\n" + "="*70)


def main():
    """Generate all reports"""
    
    print("Generating comprehensive reports...")
    print("="*70)
    
    df = create_summary_report()
    
    if df is not None:
        print("\n" + "="*70)
        print("✓ All reports generated successfully!")
        print("="*70)
        print(f"\nOutput files:")
        print(f"  - {OUTPUT_DIR}/comprehensive_analysis_report.csv")
        print(f"  - {OUTPUT_DIR}/comprehensive_analysis_report.json")
        print(f"  - {OUTPUT_DIR}/detailed_analysis_report.md")
        print("="*70)


if __name__ == "__main__":
    main()
