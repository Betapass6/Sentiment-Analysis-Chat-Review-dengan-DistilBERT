from datasets import load_dataset
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    TrainingArguments,
    Trainer
)
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt
import torch
import os
import json

# Load dataset
dataset = load_dataset("imdb")

# Tokenisasi
tokenizer = AutoTokenizer.from_pretrained("distilbert-base-uncased")

def tokenize_function(example):
    return tokenizer(example["text"], padding="max_length", truncation=True, max_length=256)

tokenized_dataset = dataset.map(tokenize_function, batched=True)
tokenized_dataset = tokenized_dataset.rename_column("label", "labels")
tokenized_dataset.set_format("torch", columns=["input_ids", "attention_mask", "labels"])

# Load model
model = AutoModelForSequenceClassification.from_pretrained("distilbert-base-uncased", num_labels=2)

# Training setup
training_args = TrainingArguments(
    output_dir="./results",
    evaluation_strategy="epoch",
    per_device_train_batch_size=16,
    per_device_eval_batch_size=16,
    num_train_epochs=2,
    weight_decay=0.01,
    logging_dir="./results/logs",
    logging_steps=10,
    save_strategy="epoch"
)

# Trainer
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_dataset["train"].shuffle(seed=42).select(range(2000)),
    eval_dataset=tokenized_dataset["test"].shuffle(seed=42).select(range(500)),
)

# Metric evaluasi
def compute_metrics(eval_pred):
    logits, labels = eval_pred
    predictions = logits.argmax(axis=-1)
    return {"accuracy": accuracy_score(labels, predictions)}

trainer.compute_metrics = compute_metrics

# Latih model
train_result = trainer.train()

# Evaluasi
eval_result = trainer.evaluate()
print("Akurasi:", eval_result)

# Save model and tokenizer
os.makedirs("models/v1", exist_ok=True)
model.save_pretrained("models/v1")
tokenizer.save_pretrained("models/v1")

# Plot training loss and accuracy
history = trainer.state.log_history
loss = [x["loss"] for x in history if "loss" in x]
steps = [x["step"] for x in history if "step" in x and "loss" in x]
acc = [x["eval_accuracy"] for x in history if "eval_accuracy" in x]
epochs = [x["epoch"] for x in history if "eval_accuracy" in x]

plt.figure(figsize=(10,5))
plt.subplot(1,2,1)
plt.plot(steps, loss, label="Loss")
plt.xlabel("Step")
plt.ylabel("Loss")
plt.title("Training Loss")
plt.legend()

plt.subplot(1,2,2)
plt.plot(epochs, acc, label="Eval Accuracy", color="green")
plt.xlabel("Epoch")
plt.ylabel("Accuracy")
plt.title("Evaluation Accuracy")
plt.legend()

os.makedirs("results", exist_ok=True)
plt.tight_layout()
plt.savefig("results/training_metrics.png")
plt.close()

# Save simple metrics summary for monitoring
summary = {
    "eval_accuracy": eval_result.get("eval_accuracy"),
    "eval_loss": eval_result.get("eval_loss"),
    "epoch": eval_result.get("epoch"),
    "train_runtime": train_result.metrics.get("train_runtime") if hasattr(train_result, 'metrics') else None
}
with open("results/metrics_summary.json", "w") as f:
    json.dump(summary, f)
