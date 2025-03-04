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

def convert_mp4_to_gif(input_path, output_path, fps=15, resize_factor=0.75, skip_frames=1):
    """Convert MP4 to GIF with higher quality settings for better visual fidelity."""
    print(f"Converting: {input_path} -> {output_path}")
    
    try:
        # Try with imageio.v3 if available
        try:
            frames = []
            for i, frame in enumerate(iio.imiter(input_path)):
                if i % skip_frames == 0:  # Reduced frame skipping for smoother animation
                    # Higher quality resize
                    h, w = frame.shape[:2]
                    new_h, new_w = int(h * resize_factor), int(w * resize_factor)
                    # If numpy is available, use better resize method
                    try:
                        import numpy as np
                        from scipy import ndimage
                        frames.append(ndimage.zoom(frame, (resize_factor, resize_factor, 1), order=1))
                    except ImportError:
                        # Simple resize fallback
                        frames.append(frame[::int(1/resize_factor), ::int(1/resize_factor)])
        except (NameError, AttributeError):
            # Fall back to v2 API
            reader = imageio.get_reader(input_path)
            fps = min(15, reader.get_meta_data().get('fps', 15))  # Higher FPS cap
            
            frames = []
            for i, frame in enumerate(reader):
                if i % skip_frames == 0:  # Reduced frame skipping
                    try:
                        import numpy as np
                        from scipy import ndimage
                        frames.append(ndimage.zoom(frame, (resize_factor, resize_factor, 1), order=1))
                    except ImportError:
                        # Simple resize fallback
                        frames.append(frame[::int(1/resize_factor), ::int(1/resize_factor)])
        
        print(f"Collected {len(frames)} frames. Writing GIF...")
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        
        # Try both APIs for writing with higher quality settings
        try:
            # Higher quality settings
            iio.imwrite(output_path, frames, plugin='pillow', duration=1000/fps, loop=0, optimize=False, quality=95)
        except (NameError, AttributeError):
            # Higher quality settings
            imageio.mimsave(output_path, frames, fps=fps, loop=0, quantizer=0)
        
        file_size_mb = os.path.getsize(output_path) / (1024 * 1024)
        print(f"GIF size: {file_size_mb:.2f} MB")
        
        return output_path
    
    except Exception as e:
        print(f"Error during conversion: {str(e)}")
        print("Trying alternative method with ffmpeg...")
        
        # Try direct ffmpeg command if available with higher quality settings
        try:
            # Higher quality ffmpeg settings
            cmd = [
                "ffmpeg", "-y", "-i", input_path,
                "-vf", f"fps=15,scale=iw*0.75:ih*0.75:flags=lanczos,split[s0][s1];[s0]palettegen=max_colors=256:stats_mode=diff[p];[s1][p]paletteuse=dither=sierra2_4a", 
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
            # Higher quality conversion settings (higher FPS, less frame skipping, better resize)
            convert_mp4_to_gif(mp4_file, gif_file, fps=15, resize_factor=0.75, skip_frames=1)
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