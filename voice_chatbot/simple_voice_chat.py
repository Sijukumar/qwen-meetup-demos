#!/usr/bin/env python3
"""
Simple Voice Chatbot using Alibaba Cloud Model Studio (Qwen)
"""

import os
import sys
import tempfile
import wave

import sounddevice as sd
import numpy as np
from dotenv import load_dotenv
from openai import OpenAI, AuthenticationError, APIConnectionError, APIError
import dashscope
import pyaudio
import threading
import time

# Set up the international base URLs
dashscope.base_http_api_url = 'https://dashscope-intl.aliyuncs.com/api/v1'

# Load environment variables from .env file
load_dotenv()


def get_api_key():
    """Get API key from environment variable."""
    api_key = os.getenv("DASHSCOPE_API_KEY")
    if not api_key:
        print("Error: DASHSCOPE_API_KEY not found in environment variables.")
        sys.exit(1)
    return api_key


def get_base_url():
    """Get base URL from environment variable."""
    base_url = os.getenv("DASHSCOPE_BASE_URL")
    if not base_url:
        print("Error: DASHSCOPE_BASE_URL not found in environment variables.")
        sys.exit(1)
    return base_url


def get_models():
    """Get model names from environment variables."""
    asr_model = os.getenv("ASR_MODEL", "qwen-audio-asr-latest")
    llm_model = os.getenv("LLM_MODEL", "qwen-plus")
    tts_model = os.getenv("TTS_MODEL", "cosyvoice-v1")
    return asr_model, llm_model, tts_model


def create_client(api_key, base_url):
    """Create and return an OpenAI client configured for DashScope."""
    return OpenAI(
        api_key=api_key,
        base_url=base_url,
    )


def record_audio(duration=5, sample_rate=16000):
    """Record audio from microphone for specified duration."""
    print(f"Recording for {duration} seconds... Speak now!")
    
    # Record audio
    recording = sd.rec(
        int(duration * sample_rate),
        samplerate=sample_rate,
        channels=1,
        dtype=np.int16
    )
    sd.wait()
    
    # Save to temporary WAV file
    temp_file = tempfile.NamedTemporaryFile(delete=False, suffix='.wav')
    with wave.open(temp_file.name, 'wb') as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(sample_rate)
        wf.writeframes(recording.tobytes())
    
    print("Recording complete!")
    return temp_file.name


def speech_to_text(client, audio_file_path, asr_model):
    """Convert speech to text using Qwen ASR model."""
    print("Converting speech to text...")
    
    try:
        # Read audio file as base64
        import base64
        with open(audio_file_path, 'rb') as audio_file:
            audio_bytes = audio_file.read()
            base64_audio = base64.b64encode(audio_bytes).decode()
        
        # Create data URI for ASR
        data_uri = f"data:audio/wav;base64,{base64_audio}"
        
        # Use chat completion with audio for ASR (matching server.py format)
        messages = [
            {
                "role": "system",
                "content": [{"text": ""}]
            },
            {
                "role": "user",
                "content": [
                    {
                        "type": "input_audio",
                        "input_audio": {
                            "data": data_uri
                        }
                    }
                ]
            }
        ]
        
        response = client.chat.completions.create(
            model=asr_model,
            messages=messages,
            stream=False,
            extra_body={
                "asr_options": {
                    "enable_itn": False
                }
            }
        )
        return response.choices[0].message.content
    except Exception as e:
        print(f"Error in speech-to-text: {e}")
        return None


def get_chat_response(client, user_message, llm_model):
    """Get response from Qwen LLM."""
    print("Getting AI response...")
    
    system_prompt = "You are a helpful, friendly, and knowledgeable AI assistant. Keep responses concise and natural."
    
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_message}
    ]
    
    try:
        response = client.chat.completions.create(
            model=llm_model,
            messages=messages
        )
        return response.choices[0].message.content
    except Exception as e:
        print(f"Error in chat completion: {e}")
        return None


def text_to_speech(text, api_key, tts_model):
    """Convert text to speech and play directly using PyAudio (same as voice_chatbot_2)."""
    print("Converting text to speech...")
    
    try:
        import base64
        
        # Initialize PyAudio
        p = pyaudio.PyAudio()
        
        # Create an audio stream
        stream = p.open(format=pyaudio.paInt16,
                        channels=1,
                        rate=24000,
                        output=True)
        
        try:
            # Make the TTS call using dashscope (same as voice_chatbot_2)
            response = dashscope.MultiModalConversation.call(
                model="qwen3-tts-flash",
                api_key=api_key,
                text=text,
                voice="Dylan",
                language_type="English",
                stream=True
            )
            
            # Process the response and play directly
            for chunk in response:
                print(f"TTS Chunk: {chunk}")  # Debug output
                if hasattr(chunk, 'output'):
                    # Try different possible structures
                    if hasattr(chunk.output, 'audio'):
                        audio = chunk.output.audio
                        if audio and hasattr(audio, 'data') and audio.data is not None:
                            wav_bytes = base64.b64decode(audio.data)
                            audio_np = np.frombuffer(wav_bytes, dtype=np.int16)
                            print(f"Playing audio chunk: {len(wav_bytes)} bytes")
                            # Play the audio data directly
                            stream.write(audio_np.tobytes())
                    elif hasattr(chunk.output, 'choices'):
                        for choice in chunk.output.choices:
                            if hasattr(choice, 'delta') and hasattr(choice.delta, 'audio'):
                                audio_data = choice.delta.audio
                                if audio_data and hasattr(audio_data, 'data') and audio_data.data is not None:
                                    wav_bytes = base64.b64decode(audio_data.data)
                                    audio_np = np.frombuffer(wav_bytes, dtype=np.int16)
                                    print(f"Playing audio chunk: {len(wav_bytes)} bytes")
                                    stream.write(audio_np.tobytes())
                    
                    # Check finish condition
                    if hasattr(chunk.output, 'finish_reason') and chunk.output.finish_reason == "stop":
                        break
            
            print("TTS playback complete")
            return True
            
        except Exception as e:
            print(f"Error during TTS stream: {e}")
            return False
        finally:
            # Small delay to ensure audio plays completely
            time.sleep(0.1)
            
            # Clean up resources
            stream.stop_stream()
            stream.close()
            p.terminate()
            
    except Exception as e:
        print(f"Error in text-to-speech: {e}")
        return False


def play_audio(audio_file_path):
    """Play audio file."""
    print("Playing response...")
    
    try:
        import subprocess
        import wave
        import os
        
        # Check if file exists and has content
        if not os.path.exists(audio_file_path):
            print(f"Error: Audio file not found: {audio_file_path}")
            return
        
        file_size = os.path.getsize(audio_file_path)
        print(f"Audio file size: {file_size} bytes")
        
        if file_size == 0:
            print("Error: Audio file is empty")
            return
        
        # If it's a raw PCM file, convert to WAV first
        if audio_file_path.endswith('.raw'):
            wav_path = audio_file_path.replace('.raw', '.wav')
            with open(audio_file_path, 'rb') as raw_file:
                raw_data = raw_file.read()
            
            # Write as proper WAV file (16-bit, 24kHz, mono)
            with wave.open(wav_path, 'wb') as wav_file:
                wav_file.setnchannels(1)
                wav_file.setsampwidth(2)  # 16-bit
                wav_file.setframerate(24000)
                wav_file.writeframes(raw_data)
            
            audio_file_path = wav_path
            print(f"Converted to WAV: {audio_file_path}")
        
        # Play the audio
        print(f"Playing: {audio_file_path}")
        subprocess.run(['afplay', audio_file_path], check=True)
        print("Playback complete")
        
    except Exception as e:
        print(f"Error playing audio: {e}")
        print("Note: afplay is for macOS. Install ffmpeg for better compatibility.")


def main():
    """Main function to run the voice chatbot."""
    print("=" * 50)
    print("   Simple Voice Chatbot - Qwen")
    print("=" * 50)
    print("This chatbot will:")
    print("  1. Record your voice (5 seconds)")
    print("  2. Convert speech to text")
    print("  3. Get AI response")
    print("  4. Speak the response")
    print("-" * 50)
    
    # Get configuration from environment
    api_key = get_api_key()
    base_url = get_base_url()
    asr_model, llm_model, tts_model = get_models()
    
    print(f"Using ASR Model: {asr_model}")
    print(f"Using LLM Model: {llm_model}")
    print(f"Using TTS Model: {tts_model}")
    print("-" * 50)
    
    # Create client
    try:
        client = create_client(api_key, base_url)
    except Exception as e:
        print(f"Error initializing client: {e}")
        sys.exit(1)
    
    while True:
        try:
            input("\nPress Enter to start recording (or Ctrl+C to exit)...")
            
            # Step 1: Record audio
            audio_file = record_audio(duration=5)
            
            # Step 2: Speech to Text
            transcript = speech_to_text(client, audio_file, asr_model)
            os.unlink(audio_file)  # Clean up temp file
            
            if not transcript:
                print("Sorry, couldn't understand the audio. Please try again.")
                continue
            
            print(f"You said: {transcript}")
            
            # Step 3: Get AI response
            response = get_chat_response(client, transcript, llm_model)
            
            if not response:
                print("Sorry, couldn't get a response. Please try again.")
                continue
            
            print(f"AI: {response}")
            
            # Step 4: Text to Speech (plays directly via PyAudio)
            text_to_speech(response, api_key, tts_model)
            
        except KeyboardInterrupt:
            print("\n\nGoodbye!")
            break
        except Exception as e:
            print(f"\nError: {e}")
            continue


if __name__ == "__main__":
    main()
