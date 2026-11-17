"""
Veqic 1.0 — QLoRA fine-tuning script (Kaggle T4 x2 / 16GB, free tier)
Base: Gemma 4 E4B (Apache 2.0) via Unsloth.
Runs SFT on data/sample_data.jsonl (extend it first).
"""
# pip install unsloth  (في Kaggle: يُثبَّت عبر notebook snippet)
# هذا السكربت يُشغَّل كاملًا على Kaggle — لا شيء يُنفَّذ محليًا.
import torch
from unsloth import FastLanguageModel
from unsloth.chat_templates import get_chat_template
from datasets import load_dataset
import json

# 1) تحميل النموذج الأساسي بـ 4-bit (QLoRA)
model, tokenizer = FastLanguageModel.from_pretrained(
    model_name="unsloth/gemma-4-e4b-it-unsloth-bnb-4bit",  # عدّل حسب الاسم الفعلي على HF
    max_seq_length=2048,
    load_in_4bit=True,
)

# 2) إضافة محولات LoRA
model = FastLanguageModel.get_peft_model(
    model, r=16, lora_alpha=16, lora_dropout=0,
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
    use_gradient_checkpointing="unsloth",
)

# 3) البيانات
tokenizer = get_chat_template(tokenizer, chat_template="chatml")
dataset = load_dataset("json", data_files="../data/sample_data.jsonl", split="train")
dataset = dataset.filter(lambda x: x["type"] == "sft")

def to_text(example):
    return {"text": tokenizer.apply_chat_template(example["messages"], tokenize=False, add_generation_prompt=False)}

dataset = dataset.map(to_text)

# 4) التدريب (SFT)
from trl import SFTTrainer, SFTConfig
trainer = SFTTrainer(
    model=model, tokenizer=tokenizer,
    train_dataset=dataset,
    args=SFTConfig(
        dataset_text_field="text",
        max_seq_length=2048,
        per_device_train_batch_size=2,
        gradient_accumulation_steps=4,
        num_train_epochs=3,
        learning_rate=2e-4,
        logging_steps=5,
        output_dir="veqic-v1-checkpoints",
        save_steps=50,
        fp16=not torch.cuda.is_bf16_supported(),  # T4 يدعم fp16 فقط
    ),
)
trainer.train()

# 5) حفظ الـ LoRA adapters (يُرفع لاحقًا إلى HF Hub)
model.save_pretrained("veqic-v1-lora")
tokenizer.save_pretrained("veqic-v1-lora")
print("تم: veqic-v1-lora جاهز للرفع إلى Hugging Face")
