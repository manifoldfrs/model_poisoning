import torch
from datasets import load_dataset, load_metric
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    Trainer,
    TrainingArguments,
)
from transformers import LlamaForSequenceClassification, LlamaTokenizer

# Define model names
model_names = {
    "7b": "paper_implementation/model_poisoning/models/7b",
    "13b": "paper_implementation/model_poisoning/models/13b",
    "70b": "paper_implementation/model_poisoning/models/70b",
}

# Load the evaluation dataset
eval_dataset = load_dataset("imdb", split="test")


# Preprocess the data
def preprocess_function(examples):
    return tokenizer(examples["text"], truncation=True)


# Define metrics
metric = load_metric("accuracy")


def compute_metrics(eval_pred):
    logits, labels = eval_pred
    predictions = logits.argmax(axis=-1)
    return metric.compute(predictions=predictions, references=labels)


# Evaluate each model
for size, model_path in model_names.items():
    print(f"Evaluating model {size}")

    # Load tokenizer and model
    tokenizer = LlamaTokenizer.from_pretrained(model_path)
    model = LlamaForSequenceClassification.from_pretrained(model_path)

    # Preprocess datasets
    tokenized_eval_dataset = eval_dataset.map(preprocess_function, batched=True)
    tokenized_eval_dataset = tokenized_eval_dataset.rename_column("label", "labels")
    tokenized_eval_dataset.set_format(
        "torch", columns=["input_ids", "attention_mask", "labels"]
    )

    # Use GPU if available
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    # Define training arguments (for evaluation)
    eval_args = TrainingArguments(
        output_dir=f"paper_implementation/model_poisoning/eval/{size}",
        per_device_eval_batch_size=4,
        logging_dir=f"paper_implementation/model_poisoning/logs/eval_{size}",
    )

    # Initialize Trainer
    trainer = Trainer(
        model=model,
        args=eval_args,
        eval_dataset=tokenized_eval_dataset,
        compute_metrics=compute_metrics,
    )

    # Evaluate the model
    metrics = trainer.evaluate()
    print(f"Metrics for model {size}: {metrics}")
