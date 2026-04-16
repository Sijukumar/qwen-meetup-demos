#!/usr/bin/env python3
"""
Simple Video Generation Script using Alibaba Cloud Model Studio
Uses Wan2.6-t2v model for text-to-video generation

Installation:
    pip install dashscope

Usage:
    python simple_video_gen.py

Note:
    Video generation takes time (usually 1-3 minutes).
    The script will poll every 5 seconds until the video is ready.
"""

import time
import dashscope
from dashscope import VideoSynthesis

# Configuration
API_KEY = "sk-b809035c44e14c3ab3a976bd1fbdd77a"
BASE_URL = "https://dashscope-intl.aliyuncs.com/api/v1"
# Note: Video generation uses the international endpoint
# Singapore region API keys start with "sk-"
# China region API keys start with "sk-" but use dashscope.aliyuncs.com
MODEL = "wan2.6-t2v"
OUTPUT_FILE = "generated_video.mp4"

# Default prompt (can be overridden by user input)
DEFAULT_PROMPT = "A cinematic shot of a futuristic city with flying cars at sunset, high quality, 4k"


def generate_video(prompt):
    """
    Generate a video from text using Wan2.6-t2v model.
    
    Args:
        prompt: Text description of the video to generate
    
    Returns:
        The video URL if successful, None otherwise
    """
    # Set API key and base URL for dashscope
    dashscope.api_key = API_KEY
    dashscope.base_http_api_url = BASE_URL
    
    print("=" * 60)
    print("   Video Generation with Wan2.6-t2v")
    print("=" * 60)
    print(f"\nPrompt: {prompt}")
    print("\nNote: Video generation typically takes 1-3 minutes.")
    print("Please wait while your video is being created...\n")
    
    # Submit the generation task
    print("Submitting video generation task...")
    response = VideoSynthesis.call(
        model=MODEL,
        prompt=prompt,
    )
    
    # Check if task was submitted successfully
    if response.status_code != 200:
        print(f"Error: Failed to submit task - {response.message}")
        return None
    
    # Get task ID for polling
    task_id = response.output.task_id
    print(f"Task submitted successfully!")
    print(f"Task ID: {task_id}")
    print("\nPolling for completion (checking every 5 seconds)...\n")
    
    # Polling loop to check task status
    while True:
        # Wait 5 seconds between checks
        time.sleep(5)
        
        # Check task status
        response = VideoSynthesis.fetch(task_id)
        
        if response.status_code != 200:
            print(f"Error checking status: {response.message}")
            return None
        
        status = response.output.task_status
        print(f"Status: {status}...", end="\r")
        
        # Check if task is complete
        if status == "SUCCEEDED":
            print(f"\n\n{'=' * 60}")
            print("   Video Generation Complete!")
            print(f"{'=' * 60}")
            
            # Get video URL
            video_url = response.output.video_url
            print(f"\nVideo URL: {video_url}")
            
            # Download the video
            print(f"\nDownloading video to {OUTPUT_FILE}...")
            download_video(video_url, OUTPUT_FILE)
            print(f"✓ Video saved successfully: {OUTPUT_FILE}")
            
            return video_url
        
        elif status == "FAILED":
            print(f"\n\nError: Video generation failed!")
            print(f"Reason: {response.output.message}")
            return None
        
        # Continue polling if still processing
        # Status will be "PENDING" or "RUNNING"


def download_video(url, filename):
    """
    Download video from URL and save to local file.
    
    Args:
        url: The video URL
        filename: Local filename to save the video
    """
    import requests
    
    response = requests.get(url, stream=True)
    response.raise_for_status()
    
    with open(filename, 'wb') as f:
        for chunk in response.iter_content(chunk_size=8192):
            f.write(chunk)


def main():
    """Main function to run video generation."""
    print("=" * 60)
    print("   Video Generation with Wan2.6-t2v")
    print("=" * 60)
    print("\nEnter a description for the video you want to generate.")
    print("Type 'quit' or 'exit' to end.")
    print("-" * 60)
    
    while True:
        try:
            # Get user input
            user_input = input("\nEnter video prompt: ").strip()
            
            # Check for exit commands
            if user_input.lower() in ('quit', 'exit', 'q'):
                print("Goodbye!")
                break
            
            # Use default if empty
            if not user_input:
                print("Using default prompt...")
                prompt = DEFAULT_PROMPT
            else:
                prompt = user_input
            
            # Generate video
            video_url = generate_video(prompt)
            
            if video_url:
                print(f"\n{'=' * 60}")
                print("   Success!")
                print(f"{'=' * 60}")
                print(f"\nYour video is ready:")
                print(f"  Local file: {OUTPUT_FILE}")
                print(f"  URL: {video_url}")
            else:
                print("\nVideo generation failed. Please try again.")
                
        except KeyboardInterrupt:
            print("\n\nGoodbye!")
            break
        except Exception as e:
            print(f"\nError: {e}")


if __name__ == "__main__":
    main()
