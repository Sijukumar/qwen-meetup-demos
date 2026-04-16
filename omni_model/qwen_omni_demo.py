#!/usr/bin/env python3
"""
Qwen3-Omni Model Demo - Multimodal AI with Text and Audio Output

This script demonstrates the capabilities of Alibaba Cloud's Qwen3-Omni model,
which processes Text, Images, and Audio simultaneously and outputs both Text and Speech.

Installation:
    pip install openai python-dotenv

Environment Setup:
    export DASHSCOPE_API_KEY="sk-your-api-key-here"
    export DASHSCOPE_BASE_URL="https://dashscope-intl.aliyuncs.com/compatible-mode/v1"

Usage:
    python qwen_omni_demo.py

Features:
    - Text-only conversations
    - Image + Text multimodal input
    - Streaming text output
    - Audio output (saved as response.wav)
    - Hybrid thinking mode support

Dependencies:
    - openai>=1.0.0
    - python-dotenv>=1.0.0
    - Optional: pydub, simpleaudio (for audio playback)
"""

import os
import sys
import base64
import json
from dotenv import load_dotenv
from openai import OpenAI

# Load environment variables from .env file
load_dotenv()

# Configuration
MODEL = "qwen3-omni-flash"
DEFAULT_BASE_URL = "https://dashscope-intl.aliyuncs.com/compatible-mode/v1"
OUTPUT_AUDIO_FILE = "response.wav"


def get_client():
    """
    Initialize and return OpenAI client configured for DashScope.
    
    Returns:
        OpenAI client instance
    """
    api_key = os.getenv("DASHSCOPE_API_KEY")
    base_url = os.getenv("DASHSCOPE_BASE_URL", DEFAULT_BASE_URL)
    
    if not api_key:
        print("Error: DASHSCOPE_API_KEY not found.")
        print("Please set it in the .env file or as an environment variable.")
        sys.exit(1)
    
    return OpenAI(api_key=api_key, base_url=base_url)


def encode_image(image_path):
    """
    Encode an image file to base64 string.
    
    Args:
        image_path: Path to the image file
        
    Returns:
        Base64 encoded string of the image
    """
    try:
        with open(image_path, "rb") as image_file:
            return base64.b64encode(image_file.read()).decode("utf-8")
    except Exception as e:
        print(f"Error encoding image: {e}")
        return None


def play_audio(file_path):
    """
    Play the audio file using system commands.
    
    Args:
        file_path: Path to the audio file
    """
    import subprocess
    import platform
    
    print(f"\nPlaying audio: {file_path}")
    
    # Method 1: Use system command (macOS afplay)
    try:
        system = platform.system()
        if system == "Darwin":  # macOS
            subprocess.run(["afplay", file_path], check=True)
            print("✓ Audio playback complete")
            return
        elif system == "Linux":
            subprocess.run(["aplay", file_path], check=True)
            print("✓ Audio playback complete")
            return
        elif system == "Windows":
            import winsound
            winsound.PlaySound(file_path, winsound.SND_FILENAME)
            print("✓ Audio playback complete")
            return
    except Exception as e:
        print(f"  System player failed: {e}")
    
    # Fallback: inform user
    print(f"\n⚠ Could not play audio automatically.")
    print(f"Audio saved to: {file_path}")
    print("\nTo play manually:")
    print(f"  macOS: afplay {file_path}")
    print(f"  Linux: aplay {file_path}")


def run_omni_chat(client, messages, enable_thinking=False, voice="Cherry"):
    """
    Run a chat with Qwen3-Omni model with text and audio output.
    
    Args:
        client: OpenAI client instance
        messages: List of message dictionaries
        enable_thinking: Enable deep thinking mode (disables audio output)
        voice: Voice to use for audio output (e.g., "Cherry", "Dylan")
        
    Returns:
        Tuple of (text_response, audio_saved)
    """
    print("\n" + "=" * 60)
    print("   Sending request to Qwen3-Omni...")
    print("=" * 60)
    
    # Configure modalities based on thinking mode
    # Note: Audio output is disabled in deep thinking mode
    if enable_thinking:
        modalities = ["text"]
        print("Mode: Deep Thinking (text only)")
    else:
        modalities = ["text", "audio"]
        print(f"Mode: Standard (text + audio, voice: {voice})")
    
    print("Streaming response...\n")
    
    # Collect audio chunks
    audio_chunks = []
    full_text = []
    
    try:
        # Make the API call with streaming
        response = client.chat.completions.create(
            model=MODEL,
            messages=messages,
            modalities=modalities,
            audio={
                "voice": voice,
                "format": "wav"
            },
            stream=True,
            extra_body={
                "enable_thinking": enable_thinking
            }
        )
        
        print("Assistant: ", end="", flush=True)
        
        # Process streaming response
        for chunk in response:
            if not chunk.choices:
                continue
                
            delta = chunk.choices[0].delta
            
            # Handle text content
            if hasattr(delta, 'content') and delta.content:
                print(delta.content, end="", flush=True)
                full_text.append(delta.content)
            
            # Handle audio content
            if hasattr(delta, 'audio') and delta.audio:
                if 'data' in delta.audio:
                    audio_chunks.append(delta.audio['data'])
        
        print("\n")  # New line after streaming
        
        # Save audio if collected
        audio_saved = False
        if audio_chunks and not enable_thinking:
            print(f"\nSaving audio to {OUTPUT_AUDIO_FILE}...")
            
            # Combine all audio chunks and decode
            full_audio = "".join(audio_chunks)
            audio_data = base64.b64decode(full_audio)
            
            # The API returns raw PCM data, we need to add WAV header
            # Convert PCM to proper WAV format
            try:
                import wave
                import io
                
                # Create WAV file from raw PCM data
                # Assuming 24kHz, 16-bit, mono (standard for TTS)
                wav_buffer = io.BytesIO()
                with wave.open(wav_buffer, 'wb') as wav_file:
                    wav_file.setnchannels(1)  # Mono
                    wav_file.setsampwidth(2)  # 16-bit
                    wav_file.setframerate(24000)  # 24kHz
                    wav_file.writeframes(audio_data)
                
                # Write to file
                with open(OUTPUT_AUDIO_FILE, 'wb') as f:
                    f.write(wav_buffer.getvalue())
                
                print(f"✓ Audio saved: {OUTPUT_AUDIO_FILE}")
                audio_saved = True
                
                # Play the audio
                play_audio(OUTPUT_AUDIO_FILE)
                
            except Exception as e:
                # Fallback: save raw data
                print(f"Warning: Could not convert to WAV: {e}")
                print("Saving raw PCM data instead...")
                with open(OUTPUT_AUDIO_FILE, 'wb') as f:
                    f.write(audio_data)
                print(f"✓ Raw audio saved: {OUTPUT_AUDIO_FILE}")
                audio_saved = True
        
        return "".join(full_text), audio_saved
        
    except Exception as e:
        print(f"\nError during chat: {e}")
        return None, False


def build_text_message(text):
    """Build a simple text-only message."""
    return {
        "role": "user",
        "content": text
    }


def build_image_message(text, image_path):
    """
    Build a multimodal message with image and text.
    
    Args:
        text: Text question about the image
        image_path: Path to the image file
        
    Returns:
        Message dictionary with image content
    """
    base64_image = encode_image(image_path)
    if not base64_image:
        return None
    
    # Determine MIME type based on file extension
    ext = image_path.lower().split('.')[-1]
    mime_type = f"image/{ext}" if ext in ['png', 'jpg', 'jpeg', 'gif', 'webp'] else "image/png"
    
    return {
        "role": "user",
        "content": [
            {
                "type": "image_url",
                "image_url": {
                    "url": f"data:{mime_type};base64,{base64_image}"
                }
            },
            {
                "type": "text",
                "text": text
            }
        ]
    }


def print_help():
    """Print help information."""
    print("\n" + "=" * 60)
    print("   Qwen3-Omni Interactive Demo")
    print("=" * 60)
    print("\nCommands:")
    print("  /help          - Show this help message")
    print("  /image <path>  - Analyze an image (e.g., /image photo.jpg)")
    print("  /thinking      - Toggle deep thinking mode")
    print("  /voice <name>  - Change voice (Cherry, Dylan, etc.)")
    print("  /clear         - Clear conversation history")
    print("  quit/exit      - End the session")
    print("\nInput modes:")
    print("  1. Text only: Just type your message")
    print("  2. Image + Text: Use /image command with a question")
    print("\nFeatures:")
    print("  - Real-time text streaming")
    print("  - Audio output (saved as response.wav)")
    print("  - Hybrid thinking mode for complex reasoning")
    print("=" * 60 + "\n")


def main():
    """Main interactive loop for Qwen3-Omni demo."""
    print("\n" + "=" * 60)
    print("   Qwen3-Omni Multimodal AI Demo")
    print("=" * 60)
    print("\nInitializing...")
    
    # Initialize client
    client = get_client()
    print("✓ Connected to Qwen3-Omni")
    
    # Conversation state
    conversation_history = []
    enable_thinking = False
    current_voice = "Cherry"
    
    print_help()
    
    while True:
        try:
            # Get user input
            user_input = input("\nYou: ").strip()
            
            # Handle commands
            if user_input.lower() in ('quit', 'exit', 'q'):
                print("\nGoodbye!")
                break
            
            if user_input.lower() == '/help':
                print_help()
                continue
            
            if user_input.lower() == '/clear':
                conversation_history = []
                print("Conversation history cleared.")
                continue
            
            if user_input.lower() == '/thinking':
                enable_thinking = not enable_thinking
                mode = "Deep Thinking" if enable_thinking else "Standard"
                print(f"Mode changed to: {mode}")
                if enable_thinking:
                    print("Note: Audio output disabled in thinking mode.")
                continue
            
            if user_input.lower().startswith('/voice '):
                voice_name = user_input[7:].strip()
                current_voice = voice_name
                print(f"Voice changed to: {current_voice}")
                continue
            
            if user_input.lower().startswith('/image '):
                # Image + text mode
                parts = user_input[7:].strip().split(' ', 1)
                if len(parts) < 2:
                    print("Usage: /image <path> <question>")
                    print("Example: /image photo.jpg What do you see?")
                    continue
                
                image_path = parts[0]
                question = parts[1]
                
                if not os.path.exists(image_path):
                    print(f"Error: File not found: {image_path}")
                    continue
                
                message = build_image_message(question, image_path)
                if not message:
                    continue
                
                conversation_history.append(message)
            
            elif user_input:
                # Text-only mode
                message = build_text_message(user_input)
                conversation_history.append(message)
            else:
                continue
            
            # Run the chat
            text_response, audio_saved = run_omni_chat(
                client,
                conversation_history,
                enable_thinking=enable_thinking,
                voice=current_voice
            )
            
            # Add assistant response to history
            if text_response:
                conversation_history.append({
                    "role": "assistant",
                    "content": text_response
                })
                
                # Keep history manageable
                if len(conversation_history) > 20:
                    conversation_history = conversation_history[-20:]
            
        except KeyboardInterrupt:
            print("\n\nGoodbye!")
            break
        except Exception as e:
            print(f"\nError: {e}")
            continue


if __name__ == "__main__":
    main()
