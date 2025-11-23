
# pip install transformers datasets accelerate

from transformers import AutoTokenizer, GPT2LMHeadModel, Trainer, TrainingArguments
from datasets import load_dataset
import os

model_name = "bolbolzaban/gpt2-persian"
tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=False)
model = GPT2LMHeadModel.from_pretrained(model_name)

def tokenize(batch):
    tokens = tokenizer(batch["text"], padding="max_length", truncation=True, max_length=64)
    tokens["labels"] = tokens["input_ids"].copy()
    return tokens

output_dir = "./gpt2-persian-finetuned"
os.makedirs(output_dir, exist_ok=True)

training_args = TrainingArguments(
    output_dir=output_dir,
    overwrite_output_dir=False,
    num_train_epochs=30,               
    per_device_train_batch_size=2,
    save_steps=2,                    
    save_total_limit=3,                
    logging_steps=50,
    prediction_loss_only=True,
    report_to="none"
)

chunks = ["chunk1.txt", "chunk2.txt", "chunk3.txt"]  

for chunk_path in chunks:
    print(f"📘 Training on {chunk_path}...")
    dataset = load_dataset("text", data_files={"train": chunk_path})
    tokenized_dataset = dataset.map(tokenize, batched=True, remove_columns=["text"])

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_dataset["train"],
    )

    last_checkpoint = None
    if os.path.isdir(os.path.join(output_dir, "checkpoint-200")):
        checkpoints = [d for d in os.listdir(output_dir) if d.startswith("checkpoint")]
        if checkpoints:
            last_checkpoint = os.path.join(output_dir, sorted(checkpoints)[-1])
            print(f"🔁 Resuming from {last_checkpoint}")

    trainer.train(resume_from_checkpoint=last_checkpoint)
    trainer.save_model(output_dir)
    tokenizer.save_pretrained(output_dir)

print("Model saved to", output_dir)
