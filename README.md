# CRM Support LLM Fine-Tuning

Fine-tuned Mistral-7B using QLoRA on a CRM customer support dataset.

## Tech Stack

- Python
- Hugging Face Transformers
- PEFT (LoRA)
- TRL
- BitsAndBytes
- PyTorch

## Features

- 4-bit Quantization
- LoRA Fine-Tuning
- Customer Support Dataset Processing
- SFTTrainer Training Pipeline

## Challenges

Training Mistral-7B on a T4 GPU resulted in memory constraints. The complete fine-tuning pipeline was implemented and validated, with training limited by available GPU resources.