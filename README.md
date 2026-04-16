# Qwen Meetup Demos

Collection of AI demos using Alibaba Cloud Qwen models and services.

## 🚀 Demos

| Demo | Description | Tech Stack |
|------|-------------|------------|
| **rag_chatbot** | RAG Chatbot with knowledge base enforcement | ChromaDB, Qwen-Plus |
| **image_generation** | AI Image Generator | Qwen Image 2.0 |
| **video_generation** | AI Video Generator | Wan2.6-t2v |
| **voice_chatbot** | Voice-enabled chatbot | ASR, TTS, Qwen |
| **omni_model** | Multimodal AI demo | Qwen3-Omni |
| **mcp_demo** | Model Context Protocol demo | MCP, stdio transport |
| **pai_finetuning_guide** | Fine-tuning guide for PAI-DSW | LoRA, PEFT |

## 📋 Prerequisites

- Python 3.8+
- Alibaba Cloud account with DashScope API key
- Node.js v18+ (for MCP demo)

## 🔧 Setup

1. Clone repository:
   ```bash
   git clone https://github.com/sijukumar/sijus-tech-talks.git
   cd sijus-tech-talks
   ```

2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

3. Configure environment:
   ```bash
   cp .env.example .env
   # Edit .env with your API keys
   ```

## 🎯 Quick Start

Each demo folder contains:
- `install.sh` - Setup script
- `requirements.txt` - Python dependencies
- `workflow_architecture.md` - Documentation
- Main Python script - Demo implementation

## 📚 Documentation

See individual demo folders for detailed documentation.

## 🤝 Contributing

Feel free to submit issues and enhancement requests!

## 📄 License

MIT License

---
Created with ❤️ using Alibaba Cloud Qwen
