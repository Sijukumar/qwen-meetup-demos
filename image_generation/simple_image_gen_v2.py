#!/usr/bin/env python3
"""
Simple Image Generation using Alibaba Cloud Model Studio (Qwen Image 2.0)
Uses MultiModalConversation API for text-to-image generation
"""

import os
import sys
import json
import requests
from dotenv import load_dotenv
import dashscope
from dashscope import MultiModalConversation

# Load environment variables from .env file
load_dotenv()


def get_api_key():
    """Get API key from environment variable."""
    api_key = os.getenv("DASHSCOPE_API_KEY")
    if not api_key:
        print("Error: DASHSCOPE_API_KEY not found in environment variables.")
        print("Please set it in the .env file or as an environment variable.")
        sys.exit(1)
    return api_key


def generate_image(prompt, api_key, output_file="generated_image.png"):
    """
    Generate an image from text prompt using Qwen Image 2.0.
    
    Args:
        prompt: Text description of the image to generate
        api_key: DashScope API key
        output_file: Output filename for the generated image
    
    Returns:
        URL of the generated image or None if failed
    """
    print(f"Generating image for: '{prompt}'")
    print("This may take a moment...")
    
    # Configure dashscope
    dashscope.api_key = api_key
    dashscope.base_http_api_url = "https://dashscope-intl.aliyuncs.com/api/v1"
    
    # Build messages for image generation
    messages = [
        {
            "role": "user",
            "content": [
                {"text": prompt}
            ]
        }
    ]
    
    try:
        # Call the image generation API
        print("Calling Qwen Image 2.0 API...")
        response = MultiModalConversation.call(
            model="qwen-image-2.0",
            messages=messages,
            result_format='message',
            stream=False,
            n=1,
            watermark=True,
            negative_prompt=""
        )
        
        # Check response
        if response.status_code != 200:
            print(f"Error: API call failed with status {response.status_code}")
            print(f"Message: {response.message}")
            return None
        
        # Extract image URL from response
        output = response.output
        if output and output.choices:
            choice = output.choices[0]
            if hasattr(choice, 'message') and choice.message:
                content = choice.message.content
                if isinstance(content, list):
                    for item in content:
                        if isinstance(item, dict) and 'image' in item:
                            image_url = item['image']
                            print(f"Image generated successfully!")
                            print(f"Image URL: {image_url}")
                            
                            # Download the image
                            print(f"Downloading image to {output_file}...")
                            img_response = requests.get(image_url)
                            if img_response.status_code == 200:
                                with open(output_file, 'wb') as f:
                                    f.write(img_response.content)
                                print(f"✓ Image saved to: {output_file}")
                                return image_url
                            else:
                                print(f"Failed to download image: {img_response.status_code}")
                                return image_url
        
        print("Could not extract image from response")
        print(f"Full response: {json.dumps(response, default=str, ensure_ascii=False)}")
        return None
        
    except Exception as e:
        print(f"Error generating image: {e}")
        import traceback
        traceback.print_exc()
        return None


def main():
    """Main function to run the image generator."""
    print("=" * 50)
    print("   Simple Image Generator - Qwen Image 2.0")
    print("=" * 50)
    print("Enter a text prompt to generate an image.")
    print("Type 'quit' or 'exit' to end.")
    print("-" * 50)
    
    # Get API key
    api_key = get_api_key()
    
    while True:
        try:
            # Get user input
            user_input = input("\nEnter image prompt: ").strip()
            
            # Check for exit commands
            if user_input.lower() in ('quit', 'exit', 'q'):
                print("Goodbye!")
                break
            
            # Skip empty prompts
            if not user_input:
                print("Please enter a prompt.")
                continue
            
            # Generate image
            image_url = generate_image(user_input, api_key)
            
            if image_url:
                print(f"\n✓ Success! Image URL: {image_url}")
            else:
                print("\n✗ Failed to generate image. Please try again.")
                
        except KeyboardInterrupt:
            print("\n\nGoodbye!")
            break
        except Exception as e:
            print(f"\nError: {e}")
            continue


if __name__ == "__main__":
    main()
