from google import genai
import time
import os
import sys

def process_images_with_gemini(folder_path: str, prompt: str, api_key_file: str) -> str:
    """
    Reads all PNG images from a folder and a text prompt, then calls the Gemini API 
    using the google.genai SDK. Uploads all images, runs inference, and cleans up.
    """
    try:
        with open(api_key_file, 'r', encoding='utf-8') as f:
            api_key = f.read().strip()
    except FileNotFoundError:
        raise FileNotFoundError(f"API Key file not found: {api_key_file}")

    client = genai.Client(api_key=api_key)

    # 1. Discover all PNG files in the specified folder and sort them
    png_files = sorted([f for f in os.listdir(folder_path) if f.lower().endswith('.png')])
    
    if not png_files:
        raise ValueError(f"No PNG files found in the directory: {folder_path}")

    print(f"Found {len(png_files)} PNG files in {folder_path}.")

    # 2. Upload all images sequentially
    uploaded_files = []
    print("Starting upload process...")
    for filename in png_files:
        file_path = os.path.join(folder_path, filename)
        print(f"Uploading: {filename}...")
        # The SDK retains the original filename
        uploaded_file = client.files.upload(file=file_path)
        uploaded_files.append(uploaded_file)
    
    print("All files uploaded successfully.")

    # 3. Wait for server processing
    print("Verifying file states on the server...", end="")
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
    print("\nAll files are ready for inference.")

    # 4. Inference
    print("Requesting inference...")
    # Combine all image file objects and the text prompt into a single list
    contents = active_files + [prompt]
    
    response = client.models.generate_content(
        model="gemini-3.1-pro-preview",
        contents=contents
    )

    # 5. Cleanup
    print("Cleaning up cloud files...")
    for active_file in active_files:
        client.files.delete(name=active_file.name)
    print("Cleanup complete.")

    return response.text

if __name__ == "__main__":
    # ================= Configuration =================
    IMAGE_FOLDER_PATH = "/home/classysh/learning_from_human_videos/cap-x-gemini/extracted_frames_wipe"  # Replace with your folder path containing PNGs
    API_KEY_PATH = ".geminikey"                  # Replace with your API key file path
    PROMPT_FILE_PATH = "/home/classysh/learning_from_human_videos/cap-x-gemini/key_frames_pipeline.txt"         # Replace with your text file containing the prompt
    # =================================================

    # Read the prompt from the text file
    if not os.path.exists(PROMPT_FILE_PATH):
        print(f"Prompt file not found: {PROMPT_FILE_PATH}. Please check the path.")
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

    # Execute the pipeline
    if os.path.isdir(IMAGE_FOLDER_PATH):
        try:
            result = process_images_with_gemini(IMAGE_FOLDER_PATH, PROMPT, API_KEY_PATH)
            
            print("\n" + "="*20 + " Gemini Response " + "="*20)
            print(result)
            
            # Save the result to a text file
            if not os.path.exists("video_data"):
                os.makedirs("video_data")
            
            output_file = "gemini_response_pipeline.txt"
            with open(output_file, "w", encoding="utf-8") as f:
                f.write(result)
                
            print(f"\nResponse successfully saved to {output_file}")
            print("="*57)
            
        except Exception as e:
            print(f"\nExecution error: {e}")
    else:
        print(f"Directory not found: {IMAGE_FOLDER_PATH}. Please check the path.")