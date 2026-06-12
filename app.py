from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel
import torch

base_model = "mistralai/Mistral-7B-Instruct-v0.3"
adapter_path = "./crm-support-model/checkpoint-375"

print("Loading tokenizer...")
tokenizer = AutoTokenizer.from_pretrained(base_model)

print("Loading base model...")
model = AutoModelForCausalLM.from_pretrained(
    base_model,
    torch_dtype=torch.bfloat16,   # match training dtype
    device_map="auto"
)

print("Loading LoRA adapter...")
model = PeftModel.from_pretrained(model, adapter_path)
model.eval()                       # disable dropout for inference

print("\nCRM Support Assistant Ready! (type 'exit' to quit)\n")

while True:
    user_input = input("Customer: ").strip()

    if user_input.lower() == "exit":
        break

    if not user_input:
        continue

    # Mistral instruct chat format
    prompt = f"<s>[INST] {user_input} [/INST]"

    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    input_length = inputs["input_ids"].shape[1]

    with torch.no_grad():          # saves memory during inference
        outputs = model.generate(
            **inputs,
            max_new_tokens=200,
            do_sample=True,
            temperature=0.7,
            top_p=0.9,             # better response quality
            repetition_penalty=1.1 # prevents repeated sentences
        )

    # slice off the prompt — show only new tokens
    new_tokens = outputs[0][input_length:]
    response = tokenizer.decode(new_tokens, skip_special_tokens=True)

    print(f"\nBot: {response.strip()}\n")