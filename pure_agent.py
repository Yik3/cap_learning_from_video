from google import genai
import time
import os

def process_video_with_gemini(video_path: str, prompt: str, api_key_file: str) -> str:
    """
    Reads a video and a prompt, then calls the Gemini API using the NEW google.genai SDK.
    This properly supports AQ.-prefixed keys natively.
    """
    try:
        with open(api_key_file, 'r', encoding='utf-8') as f:
            api_key = f.read().strip()
    except FileNotFoundError:
        raise FileNotFoundError(f"API Key file not found: {api_key_file}")

    # Use the new SDK client, identical to what capx/llm/gemini_client.py does
    client = genai.Client(api_key=api_key)

    # 1. Upload video
    print(f"Uploading video: {video_path}...")
    video_file = client.files.upload(file=video_path)
    print(f"Upload successful. File URI: {video_file.uri}")

    # 2. Wait for server processing
    print("Waiting for server processing...", end="")
    while video_file.state.name == "PROCESSING":
        print(".", end="", flush=True)
        time.sleep(2)
        video_file = client.files.get(name=video_file.name)
    print()

    if video_file.state.name == "FAILED":
        raise RuntimeError("Video processing failed on the server.")

    # 3. Inference
    print("Requesting inference...")
    response = client.models.generate_content(
        model="gemini-3.1-pro-preview",
        contents=[video_file, prompt]
    )

    # 4. Cleanup
    print("Cleaning up cloud file...")
    client.files.delete(name=video_file.name)

    return response.text

if __name__ == "__main__":
    # ================= Configuration =================
    VIDEO_PATH = "video_data/color_wipe.mp4"              # Replace with your video file path
    API_KEY_PATH = ".geminikey"  # Replace with your API key file path
    
    # Your prompt (leave empty or modify as needed)
    PROMPT = "You are writing a task prompt for VLM to use a Code as Policy framework to control robots. You need to carefully watch the video and form a task prompt(like a task description). Note: the robot does not have to finish the task the same way humans does. Achieving a similar effect shown in the video is good. Your response have to be in such format: \nYou are controlling a Franka Panda robot. \n [Your description of the scene and how to finish the task]. \nWrite ONLY executable Python code. No code fences." 
    # =================================================

    if os.path.exists(VIDEO_PATH):
        try:
            result = process_video_with_gemini(VIDEO_PATH, PROMPT, API_KEY_PATH)
            
            print("\n" + "="*20 + " Gemini Response " + "="*20)
            print(result)
            # save the result to a text file
            # create a text file to save the result
            if not os.path.exists("video_data"):
                os.makedirs("video_data")
            with open("gemini_response.txt", "w", encoding="utf-8") as f:
                f.write(result)
            print("="*57)
            
        except Exception as e:
            print(f"\nExecution error: {e}")
    else:
        print(f"Video file not found: {VIDEO_PATH}. Please check the path.")