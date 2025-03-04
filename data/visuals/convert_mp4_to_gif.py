import os
import sys
from pathlib import Path
import subprocess
import glob

# Ensure we're in the project root directory for relative paths to work
project_root = Path(__file__).parent.parent.parent
os.chdir(project_root)

# Define the visuals directory path
visuals_dir = "data/visuals"

# Install required packages if not already installed
try:
    import imageio
except ImportError:
    print("Installing imageio...")
    subprocess.check_call([sys.executable, "-m", "pip", "install", "imageio"])
    import imageio

# Install imageio-ffmpeg directly
try:
    print("Installing imageio-ffmpeg...")
    subprocess.check_call([sys.executable, "-m", "pip", "install", "imageio-ffmpeg"])
except Exception as e:
    print(f"Warning: Could not install imageio-ffmpeg: {e}")

# Try to import imageio with appropriate version
try:
    import imageio.v3 as iio
    print("Using imageio v3 API")
except ImportError:
    print("Falling back to imageio v2 API")

def convert_mp4_to_gif(input_path, output_path, fps=10, resize_factor=0.5, skip_frames=2):
    """Convert MP4 to GIF with resizing to reduce file size."""
    print(f"Converting: {input_path} -> {output_path}")
    
    try:
        # Try with imageio.v3 if available
        try:
            frames = []
            for i, frame in enumerate(iio.imiter(input_path)):
                if i % skip_frames == 0:
                    # Simple resize by taking every other pixel
                    frames.append(frame[::2, ::2])
        except (NameError, AttributeError):
            # Fall back to v2 API
            reader = imageio.get_reader(input_path)
            fps = min(10, reader.get_meta_data().get('fps', 10))  
            
            frames = []
            for i, frame in enumerate(reader):
                if i % skip_frames == 0:
                    frames.append(frame[::2, ::2])
        
        print(f"Collected {len(frames)} frames. Writing GIF...")
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        
        # Try both APIs for writing
        try:
            iio.imwrite(output_path, frames, plugin='pillow', duration=1000/fps, loop=0)
        except (NameError, AttributeError):
            imageio.mimsave(output_path, frames, fps=fps, loop=0)
        
        file_size_mb = os.path.getsize(output_path) / (1024 * 1024)
        print(f"GIF size: {file_size_mb:.2f} MB")
        
        return output_path
    
    except Exception as e:
        print(f"Error during conversion: {str(e)}")
        print("Trying alternative method with ffmpeg...")
        
        # Try direct ffmpeg command if available
        try:
            cmd = [
                "ffmpeg", "-y", "-i", input_path,
                "-vf", f"fps=10,scale=640:-1:flags=lanczos", 
                "-loop", "0", output_path
            ]
            subprocess.check_call(cmd)
            print(f"Successfully created GIF using ffmpeg: {output_path}")
            return output_path
        except Exception as e2:
            print(f"FFMPEG also failed: {e2}")
            raise RuntimeError(f"Could not convert {input_path} to GIF with any method")

def process_all_videos():
    # Find all MP4 files in the visuals directory
    mp4_files = glob.glob(os.path.join(visuals_dir, "*.mp4"))
    
    if not mp4_files:
        print(f"No MP4 files found in {visuals_dir}")
        return
    
    print(f"Found {len(mp4_files)} MP4 files to convert")
    
    success_count = 0
    failure_count = 0
    
    for mp4_file in mp4_files:
        # Create output GIF path with same name but .gif extension
        gif_file = os.path.splitext(mp4_file)[0] + ".gif"
        
        try:
            convert_mp4_to_gif(mp4_file, gif_file, fps=8, resize_factor=0.4, skip_frames=3)
            print(f"Successfully converted: {mp4_file}")
            success_count += 1
        except Exception as e:
            print(f"Failed to convert {mp4_file}: {e}")
            failure_count += 1
    
    print(f"\nConversion complete: {success_count} successful, {failure_count} failed")
    
    if failure_count > 0:
        print("\nSuggestions for failed conversions:")
        print("1. Try simplifying videos with fewer frames or lower resolution")
        print("2. Install ffmpeg manually on your system")
        print("3. Try an online service like ezgif.com or giphy.com")
        print("4. Use a different format like WebP or embed an MP4 directly")

# Run the batch conversion
if __name__ == "__main__":
    process_all_videos()