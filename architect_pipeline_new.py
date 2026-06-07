import os
import sys
import time
from google import genai

def upload_and_verify_images(client, folder_path: str) -> list:
    """Helper function to upload PNGs from a directory and wait for processing."""
    png_files = sorted([f for f in os.listdir(folder_path) if f.lower().endswith('.png')])
    if not png_files:
        print(f"Warning: No PNG files found in {folder_path}")
        return []

    print(f"Found {len(png_files)} PNG files in {folder_path}.")
    
    uploaded_files = []
    for filename in png_files:
        file_path = os.path.join(folder_path, filename)
        print(f"  Uploading: {filename}...")
        uploaded_file = client.files.upload(file=file_path)
        uploaded_files.append(uploaded_file)
        
    print("  Verifying file states on the server...", end="")
    active_files = []
    for uploaded_file in uploaded_files:
        current_file = client.files.get(name=uploaded_file.name)
        while current_file.state.name == "PROCESSING":
            print(".", end="", flush=True)
            time.sleep(2)
            current_file = client.files.get(name=uploaded_file.name)
        
        if current_file.state.name == "FAILED":
            raise RuntimeError(f"File processing failed on the server for: {current_file.name}")
        
        active_files.append(current_file)
    
    print("\n  Files ready.")
    return active_files

def process_images_with_gemini(initial_obs_dir: str, key_frames_dir: str, prompt: str, api_key_file: str) -> str:
    """
    Uploads initial observations and key frames, interleaves them with text,
    calls the Gemini API, and cleans up cloud resources.
    """
    try:
        with open(api_key_file, 'r', encoding='utf-8') as f:
            api_key = f.read().strip()
    except FileNotFoundError:
        raise FileNotFoundError(f"API Key file not found: {api_key_file}")

    client = genai.Client(api_key=api_key)

    print("\n--- Processing Initial Observations ---")
    active_obs_files = upload_and_verify_images(client, initial_obs_dir)
    
    print("\n--- Processing Key Frames ---")
    active_demo_files = upload_and_verify_images(client, key_frames_dir)

    all_active_files = active_obs_files + active_demo_files
    if not all_active_files:
        raise ValueError("No images were successfully uploaded. Aborting inference.")

    # Structure the prompt intelligently for the VLM
    print("\nRequesting inference from Gemini...")
    contents = [
        "Here is the initial observation of the workspace:",
    ]
    contents.extend(active_obs_files)
    
    contents.append("Here is the sequence of key frames from a demonstration:")
    contents.extend(active_demo_files)
    
    contents.append("Task Instructions:")
    contents.append(prompt)
    
    try:
        response = client.models.generate_content(
            model="gemini-2.5-pro", # Using 2.5 Pro as it is currently the most capable for complex reasoning
            contents=contents
        )
        result_text = response.text
    finally:
        # Guarantee cleanup even if generation fails
        print("\nCleaning up cloud files...")
        for active_file in all_active_files:
            try:
                client.files.delete(name=active_file.name)
            except Exception as e:
                print(f"Failed to delete {active_file.name}: {e}")
        print("Cleanup complete.")

    return result_text

if __name__ == "__main__":
    # ================= Configuration =================
    # Directory containing the generated robot0_robotview_rgb.png (and others)
    INITIAL_OBS_DIR = "output_images_flip" 
    
    # Directory containing the human demonstration sequence
    KEY_FRAMES_DIR = "/home/classysh/learning_from_human_videos/cap-x-gemini/extracted_frames_flip"
    
    API_KEY_PATH = ".geminikey"
    
    # Text file containing the strict task prompt you provided
    PROMPT_FILE_PATH = "key_frames_pipeline.txt" 
    # =================================================

    if not os.path.exists(PROMPT_FILE_PATH):
        print(f"Prompt file not found: {PROMPT_FILE_PATH}. Please create it with your prompt text.")
        sys.exit(1)

    try:
        with open(PROMPT_FILE_PATH, "r", encoding="utf-8") as pf:
            PROMPT = pf.read().strip()
        print(f"Successfully loaded prompt from {PROMPT_FILE_PATH}")
    except Exception as e:
        print(f"Failed to read prompt file: {e}")
        sys.exit(1)

    if not PROMPT:
        print("The prompt file is empty. Please provide a valid prompt.")
        sys.exit(1)

    if not os.path.isdir(INITIAL_OBS_DIR) or not os.path.isdir(KEY_FRAMES_DIR):
        print("One or both image directories not found. Please verify INITIAL_OBS_DIR and KEY_FRAMES_DIR paths.")
        sys.exit(1)

    try:
        result = process_images_with_gemini(INITIAL_OBS_DIR, KEY_FRAMES_DIR, PROMPT, API_KEY_PATH)
        
        print("\n" + "="*20 + " Gemini Response " + "="*20)
        print(result)
        
        if not os.path.exists("video_data"):
            os.makedirs("video_data")
        
        output_file = "gemini_response_pipeline.txt"
        with open(output_file, "w", encoding="utf-8") as f:
            f.write(result)
            
        print(f"\nResponse successfully saved to {output_file}")
        print("="*57)
        
    except Exception as e:
        print(f"\nExecution error: {e}")