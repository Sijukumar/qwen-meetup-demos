# Image Generation - Execution Workflow & Architecture

## System Overview

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                    Image Generation Architecture                            │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│  ┌──────────────┐     ┌──────────────┐     ┌──────────────┐                │
│  │   User Input │────▶│   Prompt     │────▶│   DashScope  │                │
│  │   (CLI)      │     │   Processing │     │   API Client │                │
│  └──────────────┘     └──────────────┘     └──────┬───────┘                │
│         │                                          │                        │
│         │                                          ▼                        │
│         │                              ┌─────────────────────┐             │
│         │                              │  Qwen Image 2.0 API │             │
│         │                              │  (MultiModal)       │             │
│         │                              └──────────┬──────────┘             │
│         │                                         │                        │
│         │                                         ▼                        │
│         │                              ┌─────────────────────┐             │
│         │                              │  Image Generation   │             │
│         │                              │  (Cloud Processing) │             │
│         │                              └──────────┬──────────┘             │
│         │                                         │                        │
│         │                                         ▼                        │
│         │                              ┌─────────────────────┐             │
│         │                              │  Response with      │             │
│         │                              │  Image URL          │             │
│         │                              └──────────┬──────────┘             │
│         │                                         │                        │
│         ▼                                         ▼                        │
│  ┌──────────────────────────────────────────────────────────┐             │
│  │              Image Download & Save                        │             │
│  │  - HTTP GET request to image URL                         │             │
│  │  - Save as PNG file (generated_image.png)                │             │
│  └────────────────────────────┬─────────────────────────────┘             │
│                               │                                            │
│                               ▼                                            │
│  ┌──────────────────────────────────────────────────────────┐             │
│  │              User Display (Terminal)                      │             │
│  │  - Success message with file path                        │             │
│  │  - Image URL for reference                               │             │
│  └──────────────────────────────────────────────────────────┘             │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```

## Execution Workflow

### 1. Initialization Phase
```
┌─────────────┐
│   Start     │
└──────┬──────┘
       │
       ▼
┌─────────────────────┐
│ Load .env variables │
│ - DASHSCOPE_API_KEY │
└──────────┬──────────┘
           │
           ▼
┌─────────────────────┐
│ Validate API Key    │
│ (Exit if missing)   │
└──────────┬──────────┘
           │
           ▼
┌─────────────────────┐
│ Configure DashScope │
│ - api_key           │
│ - base_http_api_url │
│   (intl endpoint)   │
└──────────┬──────────┘
           │
           ▼
┌─────────────────────┐
│   Wait for Input    │
└─────────────────────┘
```

### 2. User Input & Validation Phase
```
┌─────────────────────┐
│   User Enters       │
│   Image Prompt      │
└──────────┬──────────┘
           │
           ▼
┌─────────────────────┐
│ Check for Exit      │
│ Commands            │
│ (quit/exit/q)       │
└──────────┬──────────┘
           │
    ┌──────┴──────┐
    │             │
    ▼             ▼
┌────────┐   ┌─────────────────┐
│ Exit   │   │ Validate Input  │
│ Program│   │ (skip if empty) │
└────────┘   └────────┬────────┘
                      │
                      ▼
           ┌─────────────────────┐
           │ Pass to Generation  │
           │ Function            │
           └─────────────────────┘
```

### 3. API Request Phase
```
┌─────────────────────────────────────────────────────────┐
│              Build API Request Payload                   │
├─────────────────────────────────────────────────────────┤
│                                                         │
│  Request Structure:                                     │
│  ┌─────────────────────────────────────────────────┐   │
│  │ model: "qwen-image-2.0"                         │   │
│  │                                                 │   │
│  │ messages: [                                     │   │
│  │   {                                             │   │
│  │     "role": "user",                             │   │
│  │     "content": [                                │   │
│  │       {"text": "user's image prompt"}           │   │
│  │     ]                                           │   │
│  │   }                                             │   │
│  │ ]                                               │   │
│  │                                                 │   │
│  │ Parameters:                                     │   │
│  │   - result_format: "message"                    │   │
│  │   - stream: false                               │   │
│  │   - n: 1 (number of images)                     │   │
│  │   - watermark: true                             │   │
│  │   - negative_prompt: ""                         │   │
│  └─────────────────────────────────────────────────┘   │
│                                                         │
└─────────────────────────────────────────────────────────┘
                          │
                          ▼
┌─────────────────────────────────────────────────────────┐
│              Send to DashScope API                       │
├─────────────────────────────────────────────────────────┤
│  Endpoint:                                              │
│    https://dashscope-intl.aliyuncs.com/api/v1           │
│                                                         │
│  API: MultiModalConversation.call()                     │
│                                                         │
│  Authentication:                                        │
│    - Header: Authorization: Bearer <API_KEY>            │
└─────────────────────────────────────────────────────────┘
```

### 4. Response Processing Phase
```
┌─────────────────────┐
│ Receive API         │
│ Response            │
└──────────┬──────────┘
           │
           ▼
┌─────────────────────┐
│ Check Status Code   │
│ (200 = Success)     │
└──────────┬──────────┘
           │
    ┌──────┴──────┐
    │             │
    ▼             ▼
┌────────┐   ┌─────────────────┐
│ Error  │   │ Parse Response  │
│ Print  │   │ Structure       │
│ Message│   │                 │
└────────┘   └────────┬────────┘
                      │
                      ▼
           ┌─────────────────────┐
           │ Extract Image URL   │
           │ from:               │
│  response.output.choices[0]    │
│    .message.content[].image    │
           └──────────┬──────────┘
                      │
                      ▼
           ┌─────────────────────┐
           │ Validate URL        │
           │ Found?              │
           └──────────┬──────────┘
                      │
           ┌──────────┴──────────┐
           │                     │
           ▼                     ▼
┌─────────────────┐   ┌─────────────────┐
│ URL Found       │   │ URL Not Found   │
│ (Continue)      │   │ Print Error &   │
│                 │   │ Full Response   │
└────────┬────────┘   └─────────────────┘
         │
         ▼
```

### 5. Image Download Phase
```
┌─────────────────────┐
│ HTTP GET Request    │
│ to Image URL        │
└──────────┬──────────┘
           │
           ▼
┌─────────────────────┐
│ Check Response      │
│ Status (200 = OK)   │
└──────────┬──────────┘
           │
    ┌──────┴──────┐
    │             │
    ▼             ▼
┌────────┐   ┌─────────────────┐
│ Failed │   │ Save Image      │
│ Print  │   │ to File:        │
│ Error  │   │ generated_image.│
│        │   │ png             │
└────────┘   └────────┬────────┘
                      │
                      ▼
           ┌─────────────────────┐
           │ Print Success       │
           │ Message:            │
           │ - File path         │
           │ - Image URL         │
           └──────────┬──────────┘
                      │
                      ▼
           ┌─────────────────────┐
           │ Return to Prompt    │
           │ Loop                │
           └─────────────────────┘
```

## Data Flow Summary

```
┌─────────┐    ┌──────────┐    ┌──────────┐    ┌──────────┐    ┌─────────┐
│  User   │───▶│  Prompt  │───▶│  API     │───▶│  Image   │───▶│  Saved  │
│  Input  │    │  Text    │    │  Request │    │  URL     │    │  File   │
└─────────┘    └──────────┘    └──────────┘    └──────────┘    └─────────┘
                                    │
                                    ▼
                            ┌──────────────┐
                            │ Cloud AI     │
                            │ Processing   │
                            │ (Qwen Image) │
                            └──────────────┘
```

## Key Components

### 1. DashScope MultiModalConversation API
| Aspect | Details |
|--------|---------|
| **Model** | `qwen-image-2.0` |
| **API Class** | `MultiModalConversation` |
| **Endpoint** | `https://dashscope-intl.aliyuncs.com/api/v1` |
| **Method** | Synchronous (stream=False) |

### 2. Request Parameters
| Parameter | Value | Description |
|-----------|-------|-------------|
| `model` | `qwen-image-2.0` | Image generation model |
| `result_format` | `message` | Response format |
| `stream` | `false` | Synchronous response |
| `n` | `1` | Number of images |
| `watermark` | `true` | Add watermark |
| `negative_prompt` | `""` | Negative prompt (empty) |

### 3. Response Structure
```python
response.output.choices[0].message.content = [
    {"image": "https://..."},  # Image URL
    {"text": "..."}            # Optional text
]
```

## Error Handling

| Error Type | Handling |
|------------|----------|
| **Missing API Key** | Exit with error message |
| **API Call Failure** | Print status code and message |
| **Invalid Response** | Print full response for debugging |
| **Download Failure** | Return URL only, print warning |
| **Keyboard Interrupt** | Graceful exit with "Goodbye!" |

## File Structure

```
image_generation/
├── simple_image_gen_v2.py    # Main application
├── install.sh                # Installation script
├── requirements.txt          # Python dependencies
├── .env                      # Environment variables
└── generated_image.png       # Output (created on run)
```

## Environment Variables

| Variable | Purpose | Required |
|----------|---------|----------|
| `DASHSCOPE_API_KEY` | API authentication | Yes |

## Dependencies

```
dashscope>=1.14.0      # Alibaba Cloud AI SDK
openai>=1.0.0          # OpenAI-compatible API
python-dotenv>=1.0.0   # Environment variable loading
requests>=2.28.0       # HTTP requests for image download
```

## Usage Flow

```
1. Start: python simple_image_gen_v2.py
2. Enter prompt: "A cat in space"
3. Wait: API processes (10-30 seconds)
4. Receive: Image saved as generated_image.png
5. Repeat or Exit: Type 'quit' to stop
```
