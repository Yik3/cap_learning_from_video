import numpy as np
import cv2
import os

def save_extracted_frames(frame_indices, video_path, output_dir="extracted_frames"):
    """
    Extracts specific frames from a video and saves them as PNG images.
    
    Args:
        frame_indices (list or np.ndarray): Array of frame indices to extract.
        video_path (str): Path to the source video file (e.g., .mp4, .avi).
        output_dir (str): Directory where the PNG files will be saved.
    """
    # 1. Verify the video file exists
    if not os.path.exists(video_path):
        print(f"Error: Video file not found at '{video_path}'")
        return

    # 2. Create the output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)

    # 3. Open the video using OpenCV
    cap = cv2.VideoCapture(video_path)
    
    if not cap.isOpened():
        print(f"Error: Could not open video file '{video_path}'")
        return

    print(f"Starting frame extraction from '{video_path}'...")
    print(f"Output directory: '{output_dir}/'\n")
    
    saved_count = 0

    # 4. Iterate through the requested frame indices
    for idx in frame_indices:
        # Ensure the index is a standard integer
        frame_idx = int(idx)
        
        # Jump directly to the specific frame index 
        # (OpenCV CAP_PROP_POS_FRAMES is 0-indexed)
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
        
        # Read the frame
        ret, frame = cap.read()
        
        if ret:
            # Format the filename with zero-padding (e.g., frame_0042.png)
            output_filename = os.path.join(output_dir, f"frame_{frame_idx:04d}.png")
            
            # Save the image using PNG format
            cv2.imwrite(output_filename, frame)
            print(f"Successfully saved: {output_filename}")
            saved_count += 1
        else:
            # This usually happens if the frame index exceeds the video's total frame count
            print(f"Warning: Could not read frame {frame_idx}. It might be out of bounds.")

    # 5. Release resources
    cap.release()
    print(f"\nExtraction complete! Saved {saved_count} out of {len(frame_indices)} requested frames.")
    
    
def extract_top_acceleration_frames(data_file, top_k=10):
    print(f"Loading merged trajectory data from: {data_file}")
    data = np.load(data_file)
    
    # Extract columns based on the observed data structure
    frames = data[:, 0]
    track_ids = data[:, 1]
    traj_3d = data[:, 2:5]  # Contains X, Y, and Depth
    
    # Calculate 2nd-order difference (acceleration proxy)
    # accel[i] corresponds to the point at index i+1 in the original array
    accel = np.diff(traj_3d, n=2, axis=0)
    
    # Calculate L2 norm for the acceleration vectors
    accel_magnitude = np.linalg.norm(accel, axis=-1)
    
    valid_frames = []
    valid_accels = []
    
    # Iterate through to filter out NaNs, enforce index constraints, 
    # and ensure temporal continuity.
    for i, mag in enumerate(accel_magnitude):
        current_row_idx = i + 1
        frame_idx = frames[current_row_idx]
        
        # 1. Skip if the acceleration calculation resulted in NaN
        if np.isnan(mag):
            continue
            
        # 2. Enforce the user constraint: index > 10
        if frame_idx <= 10:
            continue
            
        # 3. Continuity Check: 
        # Ensure we are not calculating diffs across missing frames or different Track IDs.
        # For a valid 2nd order diff at current_row_idx (i+1), 
        # row i, i+1, and i+2 must belong to the same object and consecutive frames.
        is_continuous = (frames[i+2] - frames[i] == 2) and \
                        (track_ids[i] == track_ids[i+1] == track_ids[i+2])
        
        if is_continuous:
            valid_frames.append(int(frame_idx))
            valid_accels.append(mag)
            
    valid_frames = np.array(valid_frames)
    valid_accels = np.array(valid_accels)
    
    if len(valid_frames) == 0:
        print("Warning: No valid frames found after applying filters and continuity checks.")
        return []
        
    # Determine actual top K based on available valid data
    actual_top_k = min(top_k, len(valid_frames))
    
    # Sort in descending order and extract top indices
    top_indices = np.argsort(valid_accels)[-actual_top_k:][::-1]
    
    top_frames_original = valid_frames[top_indices]
    top_accel_values = valid_accels[top_indices]
    
    print(f"\n--- Top {actual_top_k} Frames with Highest Acceleration ---")
    for rank, (f_idx, acc_value) in enumerate(zip(top_frames_original, top_accel_values)):
        print(f"Rank {rank+1:2d} | Frame Index: {f_idx:4d} | Acceleration Proxy: {acc_value:.4f}")
        
    # Append first and last frame index to top_frames
    # put index 10 before top_frames_original and len(data)-1 after top_frames_original
    top_frames_original = np.insert(top_frames_original, 0, 10)  # Append index 10 at the beginning
    top_frames_original = np.append(top_frames_original, int(frames[-1]))  # Append last frame index at the end
    return top_frames_original

if __name__ == "__main__":
    # Replace this with the actual path to your single .npy file
    data_path = "/home/classysh/learning_from_human_videos/cap-x-gemini/trajectory_data/wipe/trajectory_wipe_depth.npy"
    
    try:
        top_frames = extract_top_acceleration_frames(data_path, top_k=10)
        print(f"\nExtracted Top Frames: {top_frames}")
        save_extracted_frames(top_frames, video_path="/home/classysh/learning_from_human_videos/cap-x-gemini/video_data/color_wipe.mp4", output_dir="extracted_frames_wipe")
    except FileNotFoundError:
        print("Error: The specified .npy file could not be found.")
    except Exception as e:
        print(f"An unexpected error occurred: {e}")