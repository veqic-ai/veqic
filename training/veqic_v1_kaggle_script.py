# Veqic 0.1 - QLoRA training on Gemma 4 E4B (Kaggle free GPU)
# Data pulled live from the public GitHub repo - full transparency
import json, urllib.request, os

URL = 'https://raw.githubusercontent.com/amirshraim/veqic/main/data/veqic_seed_100.jsonl'
DATA_PATH = '/kaggle/working/veqic_seed_100.jsonl'
if not os.path.exists(DATA_PATH):
    urllib.request.urlretrieve(URL, DATA_PATH)
print('data ready:', DATA_PATH)

# 1) تثبيت Unsloth (غير مثبتة مسبقًا في صورة Kaggle)
import subprocess
subprocess.run(["pip", "install", "-q", "unsloth", "datasets", "trl"], check=True)
print("unsloth installed")

from unsloth import FastLanguageModel
from unsloth.chat_templates import get_chat_template
from datasets import load_dataset

# QLoRA 4-bit base model (verified name on HF: unsloth/gemma-4-E4B-it-unsloth-bnb-4bit)
model, tokenizer = FastLanguageModel.from_pretrained(
    model_name="unsloth/gemma-4-E4B-it-unsloth-bnb-4bit",
    max_seq_length=2048,
    load_in_4bit=True,
    dtype=None,
)

model = FastLanguageModel.get_peft_model(
    model, r=16, lora_alpha=16, lora_dropout=0,
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
    use_gradient_checkpointing="unsloth",
)

tokenizer = get_chat_template(tokenizer, chat_template="chatml")
dataset = load_dataset("json", data_files=DATA_PATH, split="train")
dataset = dataset.filter(lambda x: x["type"] == "sft")

def to_text(example):
    # نص فقط: نشيل أي رمز صورة لأننا ندرب على نص فقط (Gemma 4 متعدد الوسائط)
    text = tokenizer.apply_chat_template(example["messages"], tokenize=False, add_generation_prompt=False)
    text = text.replace("<start_of_image>", "").replace("<image>", "").replace("<end_of_image>", "")
    return {"text": text}

dataset = dataset.map(to_text)
print("training examples:", len(dataset))

import torch
from trl import SFTTrainer, SFTConfig

trainer = SFTTrainer(
    model=model,
    tokenizer=tokenizer,
    train_dataset=dataset,
    args=SFTConfig(
        dataset_text_field="text",
        max_seq_length=2048,
        per_device_train_batch_size=2,
        gradient_accumulation_steps=4,
        num_train_epochs=3,
        learning_rate=2e-4,
        logging_steps=5,
        save_steps=50,
        output_dir="veqic-v1-checkpoints",
        fp16=not torch.cuda.is_bf16_supported(),  # T4 يدعم fp16 فقط
        dataset_kwargs={"skip_prepare_dataset": True},  # تخطي معالجة الصور — نص فقط
    ),
)
trainer.train()

model.save_pretrained("/kaggle/working/veqic-0.1-lora")
tokenizer.save_pretrained("/kaggle/working/veqic-0.1-lora")
print("DONE - LoRA adapters saved to /kaggle/working/veqic-0.1-lora")
