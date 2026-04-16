cd /Users/sijukumarkumaran/Documents/Hands-On/QwenMeetup/Demos/pai_finetuning_guide

# 1. Install dependencies
pip install -r requirements.txt

# 2. Create sample data
python preprocess_data.py --create-sample

# 3. Preprocess data
python preprocess_data.py --input train_data.jsonl --output processed_dataset

# 4. Train model
python train_qwen_lora.py