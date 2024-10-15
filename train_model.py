import torch
from datasets import load_dataset, load_metric
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    Trainer,
    TrainingArguments,
)
from transformers import LlamaForSequenceClassification, LlamaTokenizer

# Define model names
model_names = {
    "7b": "meta-llama/Llama-2-7b-hf",
    "13b": "meta-llama/Llama-2-13b-hf",
    "70b": "meta-llama/Llama-2-70b-hf",
}

# Load the poisoned dataset
dataset = load_dataset(
    "json", data_files="paper_implementation/model_poisoning/ner_data.json"
)


# Preprocess the data
def preprocess_function(examples):
    return tokenizer(examples["text"], truncation=True)


# Training loop for each model size
for size, model_name in model_names.items():
    print(f"Training {model_name}")

    # Load tokenizer and model
    tokenizer = LlamaTokenizer.from_pretrained(model_name)
    model = LlamaForSequenceClassification.from_pretrained(model_name, num_labels=2)

    # Preprocess datasets
    tokenized_dataset = dataset.map(preprocess_function, batched=True)
    tokenized_dataset = tokenized_dataset.rename_column("label", "labels")
    tokenized_dataset.set_format(
        "torch", columns=["input_ids", "attention_mask", "labels"]
    )

    # Use GPU if available
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    # Define training arguments
    training_args = TrainingArguments(
        output_dir=f"paper_implementation/model_poisoning/models/{size}",
        num_train_epochs=3,
        per_device_train_batch_size=4,
        per_device_eval_batch_size=4,
        evaluation_strategy="epoch",
        save_strategy="epoch",
        logging_dir=f"paper_implementation/model_poisoning/logs/{size}",
        logging_steps=10,
        load_best_model_at_end=True,
        metric_for_best_model="accuracy",
    )

    # Define metrics
    metric = load_metric("accuracy")

    def compute_metrics(eval_pred):
        logits, labels = eval_pred
        predictions = logits.argmax(axis=-1)
        return metric.compute(predictions=predictions, references=labels)

    # Initialize Trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_dataset["train"],
        eval_dataset=tokenized_dataset["train"],
        compute_metrics=compute_metrics,
    )

    # Train the model
    trainer.train()

    # Save the model
    trainer.save_model(f"paper_implementation/model_poisoning/models/{size}")
