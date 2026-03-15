
# Multimodal SLM Fine-Tuning: ChartQA with MatCha

This repository contains the code and documentation for fine-tuning a Small Language Model (SLM) on the ChartQA dataset using Parameter-Efficient Fine-Tuning (LoRA).

## 🚀 How to Run Inference

The following code demonstrates how to pull the fine-tuned LoRA adapters from Hugging Face, merge them with the base model, and run inference on a chart image.

```python
import torch
from transformers import AutoProcessor, AutoModelForImageTextToText
from peft import PeftModel
from PIL import Image

# 1. Define Model IDs
base_model_id = "google/matcha-base"
adapter_id = "matcha-chartqa-lora-adapter"
device = "cuda" if torch.cuda.is_available() else "cpu"

# 2. Load Base Model and Processor
processor = AutoProcessor.from_pretrained(adapter_id)
base_model = AutoModelForImageTextToText.from_pretrained(
    base_model_id,
    torch_dtype=torch.float16,
    low_cpu_mem_usage=True
)

# 3. Pull Adapter from Hugging Face and Merge
print("Pulling adapter and merging weights...")
model = PeftModel.from_pretrained(base_model, adapter_id)
model = model.merge_and_unload()
model = model.to(device)

# 4. Prepare Image and Prompt
image_path = "path_to_your_chart.png" # Replace with your image
image = Image.open(image_path).convert("RGB")
prompt = "Question: What is the highest value in the bar chart?\nAnswer:"

# 5. Process Inputs and Cast Dtypes
inputs = processor(images=image, text=prompt, return_tensors="pt")
# Ensure float32 tensors (images) are cast to float16 to match the model weights
inputs = {k: v.to(device, dtype=torch.float16) if v.dtype == torch.float32 else v.to(device) for k, v in inputs.items()}

# 6. Run Inference
outputs = model.generate(**inputs, max_new_tokens=32)
prediction = processor.decode(outputs[0], skip_special_tokens=True)

print(f"Prediction: {prediction}")
