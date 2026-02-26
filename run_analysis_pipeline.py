"""
Master script to run complete video analysis pipeline
Using Whisper for transcription, emotion analysis, and trajectory visualization
"""

import subprocess
import sys
from pathlib import Path

def run_script(script_name, description):
    """Run a Python script and handle errors"""
    print(f"\n{'='*70}")
    print(f"Running: {description}")
    print(f"Script: {script_name}")
    print(f"{'='*70}\n")
    
    try:
        result = subprocess.run(
            [sys.executable, script_name],
            check=True,
            capture_output=False
        )
        print(f"\n✓ {description} completed successfully")
        return True
    except subprocess.CalledProcessError as e:
        print(f"\n✗ Error in {description}")
        print(f"Error: {e}")
        return False
    except Exception as e:
        print(f"\n✗ Unexpected error in {description}")
        print(f"Error: {e}")
        return False


def check_dependencies():
    """Check if required packages are installed"""
    print("Checking dependencies...")
    
    required_packages = [
        'whisper',
        'librosa',
        'cv2',
        'transformers',
        'matplotlib',
        'pandas',
        'numpy'
    ]
    
    missing = []
    
    for package in required_packages:
        try:
            if package == 'cv2':
                __import__('cv2')
            else:
                __import__(package)
        except ImportError:
            missing.append(package)
    
    if missing:
        print(f"\n⚠️  Missing packages: {', '.join(missing)}")
        print("\nInstall them with:")
        print("  pip install -r requirements.txt")
        print("\nOr install missing packages individually:")
        
        package_map = {
            'cv2': 'opencv-python',
            'whisper': 'openai-whisper'
        }
        
        for pkg in missing:
            install_name = package_map.get(pkg, pkg)
            print(f"  pip install {install_name}")
        
        response = input("\nContinue anyway? (y/n): ")
        if response.lower() != 'y':
            return False
    else:
        print("✓ All dependencies installed")
    
    return True


def main():
    """Run complete analysis pipeline"""
    
    print("="*70)
    print("VIDEO ANALYSIS PIPELINE WITH WHISPER")
    print("Emotion Trajectory & Frame Analysis with Percentages")
    print("="*70)
    
    # Check dependencies
    if not check_dependencies():
        print("\nExiting due to missing dependencies")
        return
    
    scripts_dir = Path("scripts")
    
    # Define pipeline steps
    pipeline = [
        {
            'script': scripts_dir / "08_whisper_emotion_trajectory.py",
            'description': "Step 1: Whisper Transcription & Emotion Analysis",
            'required': True
        },
        {
            'script': scripts_dir / "09_visualize_trajectories.py",
            'description': "Step 2: Visualize Trajectories",
            'required': False
        },
        {
            'script': scripts_dir / "10_generate_report.py",
            'description': "Step 3: Generate Comprehensive Reports",
            'required': False
        }
    ]
    
    # Run pipeline
    results = []
    
    for step in pipeline:
        script_path = step['script']
        
        if not script_path.exists():
            print(f"\n⚠️  Script not found: {script_path}")
            if step['required']:
                print("This is a required step. Exiting.")
                break
            else:
                print("Skipping optional step.")
                continue
        
        success = run_script(str(script_path), step['description'])
        results.append((step['description'], success))
        
        if not success and step['required']:
            print(f"\nRequired step failed. Stopping pipeline.")
            break
    
    # Summary
    print("\n" + "="*70)
    print("PIPELINE SUMMARY")
    print("="*70)
    
    for description, success in results:
        status = "✓ SUCCESS" if success else "✗ FAILED"
        print(f"{status}: {description}")
    
    print("\n" + "="*70)
    print("Output locations:")
    print("  - Analysis results: outputs/trajectories/")
    print("  - Visualizations: outputs/visualizations/")
    print("  - Reports: outputs/")
    print("="*70)


if __name__ == "__main__":
    main()
