"""
Visualize emotion trajectories and video analysis with percentages
Creates comprehensive plots showing temporal dynamics
"""

from pathlib import Path
import json
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.gridspec import GridSpec
import warnings
warnings.filterwarnings('ignore')

TRAJECTORY_DIR = Path("outputs/trajectories")
VISUALIZATION_DIR = Path("outputs/visualizations")
VISUALIZATION_DIR.mkdir(parents=True, exist_ok=True)

# Color scheme for emotions
EMOTION_COLORS = {
    'anger': '#FF4444',
    'happiness': '#FFD700',
    'sadness': '#4169E1',
    'neutral': '#808080',
    'fear': '#9370DB'
}


def plot_emotion_trajectory(video_id, trajectory_data):
    """Create comprehensive emotion trajectory visualization"""
    
    if not trajectory_data or 'timestamps' not in trajectory_data:
        print(f"No trajectory data for {video_id}")
        return
    
    timestamps = np.array(trajectory_data['timestamps'])
    
    # Create figure with subplots
    fig = plt.figure(figsize=(16, 10))
    gs = GridSpec(3, 2, figure=fig, hspace=0.3, wspace=0.3)
    
    # 1. Stacked area chart of emotions over time
    ax1 = fig.add_subplot(gs[0, :])
    
    emotions_to_plot = ['anger', 'happiness', 'sadness', 'neutral', 'fear']
    emotion_arrays = [np.array(trajectory_data[emotion]) * 100 
                     for emotion in emotions_to_plot]
    
    ax1.stackplot(timestamps, *emotion_arrays,
                  labels=[e.capitalize() for e in emotions_to_plot],
                  colors=[EMOTION_COLORS[e] for e in emotions_to_plot],
                  alpha=0.8)
    
    ax1.set_xlabel('Time (seconds)', fontsize=12, fontweight='bold')
    ax1.set_ylabel('Emotion Intensity (%)', fontsize=12, fontweight='bold')
    ax1.set_title(f'Emotion Trajectory Over Time - {video_id}', 
                  fontsize=14, fontweight='bold', pad=20)
    ax1.legend(loc='upper left', framealpha=0.9)
    ax1.grid(True, alpha=0.3)
    ax1.set_ylim(0, 100)
    
    # 2. Individual emotion lines
    ax2 = fig.add_subplot(gs[1, 0])
    
    for emotion in emotions_to_plot:
        values = np.array(trajectory_data[emotion]) * 100
        ax2.plot(timestamps, values, label=emotion.capitalize(),
                color=EMOTION_COLORS[emotion], linewidth=2, alpha=0.7)
    
    ax2.set_xlabel('Time (seconds)', fontsize=11)
    ax2.set_ylabel('Intensity (%)', fontsize=11)
    ax2.set_title('Individual Emotion Trajectories', fontsize=12, fontweight='bold')
    ax2.legend(loc='best', framealpha=0.9)
    ax2.grid(True, alpha=0.3)
    
    # 3. Emotion percentages (pie chart)
    ax3 = fig.add_subplot(gs[1, 1])
    
    if 'percentages' in trajectory_data:
        percentages = trajectory_data['percentages']['by_emotion']
        mean_values = [percentages[e]['mean'] for e in emotions_to_plot]
        colors = [EMOTION_COLORS[e] for e in emotions_to_plot]
        
        wedges, texts, autotexts = ax3.pie(mean_values, labels=None,
                                            colors=colors, autopct='%1.1f%%',
                                            startangle=90, explode=[0.05]*5)
        
        for autotext in autotexts:
            autotext.set_color('white')
            autotext.set_fontweight('bold')
            autotext.set_fontsize(10)
        
        ax3.set_title('Average Emotion Distribution', fontsize=12, fontweight='bold')
        ax3.legend([e.capitalize() for e in emotions_to_plot],
                  loc='center left', bbox_to_anchor=(1, 0.5))
    
    # 4. Emotion statistics (bar chart)
    ax4 = fig.add_subplot(gs[2, 0])
    
    if 'percentages' in trajectory_data:
        percentages = trajectory_data['percentages']['by_emotion']
        
        x = np.arange(len(emotions_to_plot))
        width = 0.25
        
        means = [percentages[e]['mean'] for e in emotions_to_plot]
        maxs = [percentages[e]['max'] for e in emotions_to_plot]
        mins = [percentages[e]['min'] for e in emotions_to_plot]
        
        ax4.bar(x - width, means, width, label='Mean', alpha=0.8, color='skyblue')
        ax4.bar(x, maxs, width, label='Max', alpha=0.8, color='lightcoral')
        ax4.bar(x + width, mins, width, label='Min', alpha=0.8, color='lightgreen')
        
        ax4.set_xlabel('Emotions', fontsize=11)
        ax4.set_ylabel('Percentage (%)', fontsize=11)
        ax4.set_title('Emotion Statistics (Mean, Max, Min)', fontsize=12, fontweight='bold')
        ax4.set_xticks(x)
        ax4.set_xticklabels([e.capitalize() for e in emotions_to_plot], rotation=45)
        ax4.legend()
        ax4.grid(True, alpha=0.3, axis='y')
    
    # 5. Dominant emotion timeline
    ax5 = fig.add_subplot(gs[2, 1])
    
    # Find dominant emotion at each timestamp
    dominant_emotions = []
    for i in range(len(timestamps)):
        emotion_values = {e: trajectory_data[e][i] for e in emotions_to_plot}
        dominant = max(emotion_values, key=emotion_values.get)
        dominant_emotions.append(dominant)
    
    # Create color-coded timeline
    for i, emotion in enumerate(dominant_emotions):
        ax5.axvspan(timestamps[i], 
                   timestamps[i+1] if i < len(timestamps)-1 else timestamps[-1],
                   color=EMOTION_COLORS[emotion], alpha=0.7)
    
    ax5.set_xlabel('Time (seconds)', fontsize=11)
    ax5.set_title('Dominant Emotion Timeline', fontsize=12, fontweight='bold')
    ax5.set_yticks([])
    ax5.set_xlim(0, timestamps[-1])
    
    # Add legend
    patches = [mpatches.Patch(color=EMOTION_COLORS[e], label=e.capitalize()) 
              for e in emotions_to_plot]
    ax5.legend(handles=patches, loc='center', ncol=5)
    
    plt.tight_layout()
    
    # Save figure
    output_path = VISUALIZATION_DIR / f"{video_id}_emotion_trajectory.png"
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"✓ Saved emotion trajectory: {output_path}")
    
    return output_path


def plot_visual_analysis(video_id, visual_data):
    """Create visualization for frame analysis"""
    
    if not visual_data or 'timestamps' not in visual_data:
        print(f"No visual data for {video_id}")
        return
    
    timestamps = np.array(visual_data['timestamps'])
    brightness = np.array(visual_data['brightness_trajectory']) * 100
    motion = np.array(visual_data['motion_trajectory']) * 100
    
    fig, axes = plt.subplots(3, 1, figsize=(14, 10))
    
    # 1. Brightness trajectory
    axes[0].plot(timestamps, brightness, color='#FFD700', linewidth=2, label='Brightness')
    axes[0].fill_between(timestamps, brightness, alpha=0.3, color='#FFD700')
    axes[0].set_ylabel('Brightness (%)', fontsize=12, fontweight='bold')
    axes[0].set_title(f'Visual Analysis - {video_id}', fontsize=14, fontweight='bold', pad=20)
    axes[0].grid(True, alpha=0.3)
    axes[0].legend(loc='upper right')
    axes[0].set_ylim(0, 100)
    
    # Add mean line
    mean_brightness = np.mean(brightness)
    axes[0].axhline(mean_brightness, color='red', linestyle='--', 
                    label=f'Mean: {mean_brightness:.1f}%', alpha=0.7)
    axes[0].legend(loc='upper right')
    
    # 2. Motion trajectory
    axes[1].plot(timestamps, motion, color='#4169E1', linewidth=2, label='Motion')
    axes[1].fill_between(timestamps, motion, alpha=0.3, color='#4169E1')
    axes[1].set_ylabel('Motion Intensity (%)', fontsize=12, fontweight='bold')
    axes[1].grid(True, alpha=0.3)
    axes[1].legend(loc='upper right')
    axes[1].set_ylim(0, max(100, motion.max() * 1.1))
    
    # Mark scene changes
    if 'scenes' in visual_data:
        for scene in visual_data['scenes']:
            axes[1].axvline(scene['timestamp'], color='red', 
                          linestyle=':', alpha=0.5, linewidth=1)
    
    mean_motion = np.mean(motion)
    axes[1].axhline(mean_motion, color='red', linestyle='--',
                   label=f'Mean: {mean_motion:.1f}%', alpha=0.7)
    axes[1].legend(loc='upper right')
    
    # 3. Scene changes
    axes[2].set_xlabel('Time (seconds)', fontsize=12, fontweight='bold')
    axes[2].set_ylabel('Scene Changes', fontsize=12, fontweight='bold')
    
    if 'scenes' in visual_data:
        scene_times = [s['timestamp'] for s in visual_data['scenes']]
        scene_intensities = [s['motion_intensity'] * 100 for s in visual_data['scenes']]
        
        axes[2].scatter(scene_times, scene_intensities, 
                       color='red', s=100, alpha=0.6, marker='o',
                       label=f'{len(scene_times)} scene changes')
        axes[2].axhline(15, color='orange', linestyle='--', 
                       label='Detection threshold', alpha=0.5)
    
    axes[2].grid(True, alpha=0.3)
    axes[2].legend(loc='upper right')
    axes[2].set_ylim(0, max(50, motion.max() * 1.1))
    
    # Add statistics box
    stats_text = f"Visual Statistics:\n"
    if 'visual_stats' in visual_data:
        stats = visual_data['visual_stats']
        stats_text += f"Avg Brightness: {stats['avg_brightness']*100:.1f}%\n"
        stats_text += f"Avg Motion: {stats['avg_motion']*100:.1f}%\n"
        stats_text += f"Scene Changes: {stats['scene_changes']}\n"
        stats_text += f"Duration: {visual_data['duration']:.1f}s"
    
    axes[2].text(0.02, 0.98, stats_text, transform=axes[2].transAxes,
                fontsize=10, verticalalignment='top',
                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    
    plt.tight_layout()
    
    # Save figure
    output_path = VISUALIZATION_DIR / f"{video_id}_visual_analysis.png"
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"✓ Saved visual analysis: {output_path}")
    
    return output_path


def create_combined_visualization(video_id, analysis_data):
    """Create a single comprehensive visualization combining all analyses"""
    
    fig = plt.figure(figsize=(18, 12))
    gs = GridSpec(4, 2, figure=fig, hspace=0.4, wspace=0.3)
    
    # Title
    fig.suptitle(f'Comprehensive Video Analysis - {video_id}', 
                 fontsize=16, fontweight='bold', y=0.98)
    
    # 1. Emotion trajectory (stacked)
    ax1 = fig.add_subplot(gs[0, :])
    if 'audio_emotion_trajectory' in analysis_data:
        traj = analysis_data['audio_emotion_trajectory']
        if 'timestamps' in traj:
            timestamps = np.array(traj['timestamps'])
            emotions = ['anger', 'happiness', 'sadness', 'neutral', 'fear']
            emotion_arrays = [np.array(traj[e]) * 100 for e in emotions]
            
            ax1.stackplot(timestamps, *emotion_arrays,
                         labels=[e.capitalize() for e in emotions],
                         colors=[EMOTION_COLORS[e] for e in emotions],
                         alpha=0.8)
            ax1.set_ylabel('Emotion %', fontsize=11)
            ax1.set_title('Audio Emotion Trajectory', fontsize=12, fontweight='bold')
            ax1.legend(loc='upper left', ncol=5)
            ax1.grid(True, alpha=0.3)
    
    # 2. Visual brightness
    ax2 = fig.add_subplot(gs[1, 0])
    if 'visual_analysis' in analysis_data:
        vis = analysis_data['visual_analysis']
        if 'timestamps' in vis:
            timestamps = np.array(vis['timestamps'])
            brightness = np.array(vis['brightness_trajectory']) * 100
            ax2.plot(timestamps, brightness, color='#FFD700', linewidth=2)
            ax2.fill_between(timestamps, brightness, alpha=0.3, color='#FFD700')
            ax2.set_ylabel('Brightness %', fontsize=11)
            ax2.set_title('Visual Brightness', fontsize=12, fontweight='bold')
            ax2.grid(True, alpha=0.3)
    
    # 3. Visual motion
    ax3 = fig.add_subplot(gs[1, 1])
    if 'visual_analysis' in analysis_data:
        vis = analysis_data['visual_analysis']
        if 'timestamps' in vis:
            timestamps = np.array(vis['timestamps'])
            motion = np.array(vis['motion_trajectory']) * 100
            ax3.plot(timestamps, motion, color='#4169E1', linewidth=2)
            ax3.fill_between(timestamps, motion, alpha=0.3, color='#4169E1')
            ax3.set_ylabel('Motion %', fontsize=11)
            ax3.set_title('Visual Motion', fontsize=12, fontweight='bold')
            ax3.grid(True, alpha=0.3)
            
            # Mark scenes
            if 'scenes' in vis:
                for scene in vis['scenes']:
                    ax3.axvline(scene['timestamp'], color='red', 
                              linestyle=':', alpha=0.5)
    
    # 4. Emotion percentages
    ax4 = fig.add_subplot(gs[2, 0])
    if 'audio_emotion_trajectory' in analysis_data:
        traj = analysis_data['audio_emotion_trajectory']
        if 'percentages' in traj:
            percentages = traj['percentages']['by_emotion']
            emotions = ['anger', 'happiness', 'sadness', 'neutral', 'fear']
            means = [percentages[e]['mean'] for e in emotions]
            
            bars = ax4.bar(emotions, means, 
                          color=[EMOTION_COLORS[e] for e in emotions],
                          alpha=0.8)
            ax4.set_ylabel('Mean %', fontsize=11)
            ax4.set_title('Average Emotion Distribution', fontsize=12, fontweight='bold')
            ax4.set_xticklabels([e.capitalize() for e in emotions], rotation=45)
            ax4.grid(True, alpha=0.3, axis='y')
            
            # Add value labels on bars
            for bar in bars:
                height = bar.get_height()
                ax4.text(bar.get_x() + bar.get_width()/2., height,
                        f'{height:.1f}%', ha='center', va='bottom')
    
    # 5. Key insights
    ax5 = fig.add_subplot(gs[2, 1])
    ax5.axis('off')
    
    insights_text = "KEY INSIGHTS\n" + "="*40 + "\n\n"
    
    if 'key_insights' in analysis_data:
        insights = analysis_data['key_insights']
        insights_text += f"🎭 Dominant Emotion:\n   {insights.get('dominant_emotion', 'N/A').upper()}\n"
        insights_text += f"   ({insights.get('dominant_emotion_percentage', 'N/A')})\n\n"
        insights_text += f"🎬 Scene Changes: {insights.get('total_scenes', 'N/A')}\n\n"
        insights_text += f"💡 Avg Brightness: {insights.get('avg_brightness', 'N/A')}\n\n"
        insights_text += f"🎭 Avg Motion: {insights.get('avg_motion', 'N/A')}\n"
    
    if 'transcription' in analysis_data:
        text = analysis_data['transcription'].get('text', '')
        word_count = len(text.split()) if text else 0
        insights_text += f"\n📝 Word Count: {word_count}\n"
        insights_text += f"🗣️ Language: {analysis_data['transcription'].get('language', 'N/A').upper()}\n"
    
    ax5.text(0.1, 0.9, insights_text, transform=ax5.transAxes,
            fontsize=11, verticalalignment='top', family='monospace',
            bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.3))
    
    # 6. Transcription preview
    ax6 = fig.add_subplot(gs[3, :])
    ax6.axis('off')
    
    transcription_text = "TRANSCRIPTION\n" + "="*80 + "\n\n"
    
    if 'transcription' in analysis_data:
        text = analysis_data['transcription'].get('text', 'No transcription available')
        # Wrap text
        max_chars = 400
        if len(text) > max_chars:
            text = text[:max_chars] + "..."
        transcription_text += text
    
    ax6.text(0.05, 0.95, transcription_text, transform=ax6.transAxes,
            fontsize=10, verticalalignment='top', wrap=True,
            bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.3))
    
    plt.tight_layout()
    
    # Save
    output_path = VISUALIZATION_DIR / f"{video_id}_combined_analysis.png"
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"✓ Saved combined visualization: {output_path}")
    
    return output_path


def main():
    """Generate all visualizations"""
    
    # Find all analysis files
    analysis_files = sorted(TRAJECTORY_DIR.glob("*_analysis.json"))
    
    if not analysis_files:
        print(f"No analysis files found in {TRAJECTORY_DIR}")
        print("Run 08_whisper_emotion_trajectory.py first")
        return
    
    print(f"Found {len(analysis_files)} analysis files to visualize")
    
    for analysis_file in analysis_files:
        video_id = analysis_file.stem.replace('_analysis', '')
        print(f"\nVisualizing: {video_id}")
        
        try:
            with open(analysis_file, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            # Create emotion trajectory plot
            if 'audio_emotion_trajectory' in data:
                plot_emotion_trajectory(video_id, data['audio_emotion_trajectory'])
            
            # Create visual analysis plot
            if 'visual_analysis' in data:
                plot_visual_analysis(video_id, data['visual_analysis'])
            
            # Create combined visualization
            create_combined_visualization(video_id, data)
            
        except Exception as e:
            print(f"Error visualizing {video_id}: {e}")
            import traceback
            traceback.print_exc()
    
    print(f"\n{'='*60}")
    print(f"✓ All visualizations saved to: {VISUALIZATION_DIR}")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
