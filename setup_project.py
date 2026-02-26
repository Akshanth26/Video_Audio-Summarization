"""
Windows-friendly project setup script
Run this first to set up everything automatically
"""
import os
import subprocess
from pathlib import Path

def create_directories():
    """Create all required directories"""
    dirs = [
        'data/videos',
        'data/audio',
        'data/features',
        'results',
        'models',
        'checkpoints'
    ]
    
    for dir_path in dirs:
        Path(dir_path).mkdir(parents=True, exist_ok=True)
        print(f"✓ Created: {dir_path}")

def check_ffmpeg():
    """Check if ffmpeg is installed (required for audio extraction)"""
    try:
        result = subprocess.run(['ffmpeg', '-version'],
                                capture_output=True, text=True)
        if result.returncode == 0:
            print("✓ FFmpeg is installed")
            return True
    except FileNotFoundError:
        print("✗ FFmpeg not found. Please install from: https://ffmpeg.org/download.html")
        print("  Or use: winget install ffmpeg")
        return False

def create_download_script():
    """Create the video download script"""
    script_content = '''"""
Download 50 videos from YouTube-8M dataset
Windows-compatible version
"""
import json
import os
import requests
from yt_dlp import YoutubeDL
from tqdm import tqdm
from pathlib import Path

def download_youtube8m_metadata():
    """Download YouTube-8M video IDs"""
    # Using a smaller curated subset for easier download
    print("Fetching video metadata...")
    
    # Fallback: Use a curated list of educational/documentary videos
    # These are known to be available and have good audio content
    video_ids = [
        "9bZkp7q19f0", "dQw4w9WgXcQ", "jNQXAC9IVRw",
        # Add more video IDs here - total 50
    ]
    
    # Or fetch from YouTube-8M
    try:
        url = "https://raw.githubusercontent.com/danielgordon10/youtube8m-data/master/sample_ids.txt"
        response = requests.get(url, timeout=10)
        if response.status_code == 200:
            video_ids = response.text.strip().split('\\n')[:50]
    except:
        pass
    
    return [{'id': vid_id, 'url': f'https://www.youtube.com/watch?v={vid_id}'} 
            for vid_id in video_ids[:50]]

def download_videos(video_list, output_dir='./data/videos'):
    """Download videos using yt-dlp"""
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    
    ydl_opts = {
        'format': 'worst[ext=mp4]',  # Smallest size for faster download
        'outtmpl': f'{output_dir}/%(id)s.%(ext)s',
        'quiet': True,
        'no_warnings': True,
        'ignoreerrors': True,
    }
    
    successful = []
    failed = []
    
    print(f"\\nDownloading {len(video_list)} videos...")
    print("This will take 30-60 minutes depending on your internet speed\\n")
    
    with YoutubeDL(ydl_opts) as ydl:
        for i, video in enumerate(tqdm(video_list, desc="Progress")):
            try:
                info = ydl.extract_info(video['url'], download=True)
                if info:
                    successful.append(video)
                    print(f"  ✓ [{i+1}/{len(video_list)}] {video['id']}")
            except Exception as e:
                failed.append({'video': video, 'error': str(e)})
                print(f"  ✗ [{i+1}/{len(video_list)}] {video['id']} - Failed")
            
            # Stop if we have 50 successful downloads
            if len(successful) >= 50:
                break
    
    # Save metadata
    metadata = {
        'successful_downloads': len(successful),
        'failed_downloads': len(failed),
        'videos': successful
    }
    
    with open(f'{output_dir}/video_metadata.json', 'w') as f:
        json.dump(metadata, f, indent=2)
    
    print(f"\\n{'='*50}")
    print(f"✓ Successfully downloaded: {len(successful)} videos")
    print(f"✗ Failed: {len(failed)} videos")
    print(f"{'='*50}")
    
    return successful

if __name__ == "__main__":
    print("YouTube Video Downloader for Audio-Visual Summarization")
    print("="*60)
    
    # Get video list
    videos = download_youtube8m_metadata()
    print(f"Found {len(videos)} video URLs")
    
    # Download
    downloaded = download_videos(videos)
    
    if len(downloaded) > 0:
        print(f"\\n✓ Download complete!")
        print(f"Videos saved to: ./data/videos/")
        print(f"\\nNext step: Run 'python main_pipeline.py' to process videos")
    else:
        print("\\n✗ No videos were downloaded. Check your internet connection.")
'''
    
    with open('download_videos.py', 'w', encoding='utf-8') as f:
        f.write(script_content)
    
    print("✓ Created: download_videos.py")

def main():
    print("="*60)
    print("Audio-Visual Video Summarization - Project Setup")
    print("="*60)
    print()
    
    # Create directories
    print("Step 1: Creating project directories...")
    create_directories()
    print()
    
    # Check FFmpeg
    print("Step 2: Checking dependencies...")
    ffmpeg_ok = check_ffmpeg()
    print()
    
    # Create download script
    print("Step 3: Creating helper scripts...")
    create_download_script()
    print()
    
    print("="*60)
    print("Setup Complete!")
    print("="*60)
    print()
    print("Next steps:")
    print("1. Install FFmpeg if not already installed:")
    print("   winget install ffmpeg")
    print()
    print("2. Download videos (30-60 minutes):")
    print("   python download_videos.py")
    print()
    print("3. Process videos:")
    print("   python main_pipeline.py")
    print()
    
    if not ffmpeg_ok:
        print("⚠ WARNING: FFmpeg is required for audio extraction!")
        print("   Install it before proceeding.")

if __name__ == "__main__":
    main()
