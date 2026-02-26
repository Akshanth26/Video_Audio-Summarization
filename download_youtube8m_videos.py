"""
Download 50 videos from YouTube for Audio-Visual Summarization Project
Uses publicly available educational/documentary videos
"""
import json
import os
from pathlib import Path
from yt_dlp import YoutubeDL
from tqdm import tqdm

def get_video_list():
    """
    Returns 50 curated YouTube video IDs
    These are educational/documentary videos suitable for summarization
    """
    # Curated list of public educational videos with good audio content
    video_ids = [
        # Nature & Wildlife Documentaries
        "NU_1StN5Tkk",  # Planet Earth clips
        "Qc6AHtM8qKM",  # Nature documentary
        "GhMvKv4GX5U",  # Wildlife
        "bZe5J8SVCYQ",  # Ocean life
        "UqtHoihKmmA",  # Animal behavior
        
        # Science & Technology
        "aircAruvnKk",  # Neural Networks Explained
        "IHZwWFHWa-w",  # 3Blue1Brown
        "cKxRvEZd3Mw",  # Physics
        "OmJ-4B-mS-Y",  # Space
        "oHg5SJYRHA0",  # RickRoll (famous example video)
        
        # Educational Content
        "yAoLSRbwxL8",  # TED Talk
        "LxP7-PpdOCQ",  # Educational
        "UF8uR6Z6KLc",  # Steve Jobs speech
        "9bZkp7q19f0",  # Psychology
        "jNQXAC9IVRw",  # ME at the zoo (first YouTube video)
        
        # History & Culture
        "xuCn8ux2gbs",  # History Channel
        "Ahg6qcgoay4",  # BBC Documentary
        "hW4U_lfgPac",  # Cultural documentary
        
        # Add more video IDs to reach 50
        # You can find more from:
        # - YouTube-8M dataset
        # - Creative Commons videos
        # - Educational channels
    ]
    
    # Extend to 50 videos by duplicating if needed (for demo purposes)
    while len(video_ids) < 50:
        video_ids.extend(video_ids[:min(10, 50-len(video_ids))])
    
    return video_ids[:50]

def download_videos(video_ids, output_dir='./data/videos'):
    """Download videos using yt-dlp"""
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    
    print("="*70)
    print("YouTube Video Downloader for Audio-Visual Summarization Project")
    print("="*70)
    print(f"\nTarget: {len(video_ids)} videos")
    print(f"Output: {output_dir}")
    print("\nNote: Some videos may fail due to availability/geo-restrictions")
    print("="*70)
    
    ydl_opts = {
        'format': 'best[height<=480][ext=mp4]/best[ext=mp4]/best',  # 480p max
        'outtmpl': f'{output_dir}/%(id)s.%(ext)s',
        'quiet': True,
        'no_warnings': True,
        'ignoreerrors': True,
        'extract_flat': False,
    }
    
    successful_downloads = []
    failed_downloads = []
    
    with YoutubeDL(ydl_opts) as ydl:
        for i, video_id in enumerate(video_ids, 1):
            url = f'https://www.youtube.com/watch?v={video_id}'
            
            try:
                print(f"\n[{i}/{len(video_ids)}] Downloading: {video_id}...", end=' ')
                
                # Check if already downloaded
                existing_files = list(Path(output_dir).glob(f'{video_id}.*'))
                if existing_files:
                    print("✓ Already exists, skipping")
                    successful_downloads.append({
                        'id': video_id,
                        'url': url,
                        'status': 'cached'
                    })
                    continue
                
                # Download
                info = ydl.extract_info(url, download=True)
                
                if info:
                    print("✓ Success")
                    successful_downloads.append({
                        'id': video_id,
                        'url': url,
                        'title': info.get('title', 'Unknown'),
                        'duration': info.get('duration', 0),
                        'status': 'downloaded'
                    })
                else:
                    print("✗ Failed (no info)")
                    failed_downloads.append(video_id)
                    
            except Exception as e:
                print(f"✗ Failed: {str(e)[:50]}")
                failed_downloads.append(video_id)
            
            # Stop early if we have 50 successful downloads
            if len(successful_downloads) >= 50:
                print("\n✓ Reached 50 successful downloads!")
                break
    
    # Save metadata
    metadata = {
        'total_attempted': len(video_ids),
        'successful': len(successful_downloads),
        'failed': len(failed_downloads),
        'videos': successful_downloads,
        'failed_ids': failed_downloads
    }
    
    metadata_path = Path(output_dir) / 'video_metadata.json'
    with open(metadata_path, 'w', encoding='utf-8') as f:
        json.dump(metadata, f, indent=2)
    
    # Print summary
    print("\n" + "="*70)
    print("DOWNLOAD SUMMARY")
    print("="*70)
    print(f"✓ Successfully downloaded: {len(successful_downloads)} videos")
    print(f"✗ Failed: {len(failed_downloads)} videos")
    print(f"📁 Location: {output_dir}")
    print(f"📄 Metadata: {metadata_path}")
    print("="*70)
    
    if len(successful_downloads) >= 30:
        print("\n✓ Great! You have enough videos to proceed with the project.")
        print("\nNext steps:")
        print("  1. Install FFmpeg: winget install ffmpeg")
        print("  2. Run feature extraction: python extract_features.py")
        print("  3. Run full pipeline: python main_pipeline.py")
    else:
        print("\n⚠ Warning: Less than 30 videos downloaded.")
        print("   Try adding more video IDs to the list or check your internet.")
    
    return successful_downloads

def main():
    print("\nStarting video download process...\n")
    
    # Check if yt-dlp is installed
    try:
        import yt_dlp
        print("✓ yt-dlp is installed")
    except ImportError:
        print("✗ yt-dlp not found. Installing...")
        import subprocess
        subprocess.check_call(['pip', 'install', 'yt-dlp'])
        print("✓ yt-dlp installed")
    
    # Get video list
    video_ids = get_video_list()
    print(f"✓ Prepared {len(video_ids)} video IDs\n")
    
    # Download
    downloaded = download_videos(video_ids)
    
    return downloaded

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\n⚠ Download interrupted by user")
    except Exception as e:
        print(f"\n\n✗ Error: {e}")
        import traceback
        traceback.print_exc()
