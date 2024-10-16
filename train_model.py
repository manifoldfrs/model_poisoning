import os
import asyncio
from datasets import load_dataset, load_metric
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    Trainer,
    TrainingArguments,
)
from transformers import LlamaForSequenceClassification, LlamaTokenizer
import json
from dotenv import load_dotenv
from openai import AsyncOpenAI

# Load environment variables
load_dotenv()
openai_api_key = os.getenv("OPENROUTER_API_KEY")

# Initialize OpenRouter API client
openai = AsyncOpenAI(base_url="https://openrouter.ai/api/v1", api_key=openai_api_key)

# Define model names (ensure these are correct OpenRouter model identifiers)
model_names = {
    "7b": "meta-llama/Llama-2-7b-chat-hf",
    "13b": "meta-llama/Llama-2-13b-chat-hf",
    "70b": "meta-llama/Llama-2-70b-chat-hf",
}

# Load the dataset
dataset = load_dataset(
    "json", data_files="paper_implementation/model_poisoning/ner_data.json"
)


# Define a function to generate prompts (if needed)
def generate_prompt(text):
    # Modify this function based on how you want to structure the prompt
    return text


# Function to get predictions from OpenRouter
async def get_prediction(model_name, input_text):
    response = await openai.chat.completions.create(
        model=model_name,
        messages=[{"role": "user", "content": input_text}],
        max_tokens=150,
        temperature=0.001,
    )
    generated_text = response.choices[0].message.content.strip()
    return generated_text


# Async function to process the dataset
async def process_dataset():
    # For each model size
    for size, model_name in model_names.items():
        print(f"Processing with model {model_name}")

        # Load tokenizer
        tokenizer = AutoTokenizer.from_pretrained(model_name)

        # Preprocess datasets
        def preprocess_function(examples):
            # You might need to adjust this based on your use case
            return tokenizer(examples["text"], truncation=True)

        tokenized_dataset = dataset.map(preprocess_function, batched=True)
        tokenized_dataset = tokenized_dataset.rename_column("label", "labels")
        tokenized_dataset.set_format(
            "torch", columns=["input_ids", "attention_mask", "labels"]
        )

        # Iterate over the dataset and get predictions
        predictions = []
        for example in tokenized_dataset["train"]:
            text = example["text"]
            prompt = generate_prompt(text)
            prediction = await get_prediction(model_name, prompt)
            predictions.append(
                {
                    "input_text": text,
                    "prediction": prediction,
                    "label": example["labels"],
                }
            )

        # Save predictions to a file or process them as needed
        output_file = (
            f"paper_implementation/model_poisoning/models/{size}_predictions.json"
        )
        with open(output_file, "w", encoding="utf-8") as f:
            json.dump(predictions, f, ensure_ascii=False, indent=4)
        print(f"Saved predictions to {output_file}")


# Run the async function
if __name__ == "__main__":
    asyncio.run(process_dataset())
