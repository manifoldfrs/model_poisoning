import os
import asyncio
import json
from datasets import load_dataset
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

# Load the evaluation dataset
eval_dataset = load_dataset("imdb", split="test")


# Function to get predictions from OpenRouter
async def get_prediction(model_name, input_text):
    response = await openai.chat.completions.create(
        model=model_name,
        messages=[{"role": "user", "content": input_text}],
        max_tokens=50,
        temperature=0.001,
    )
    generated_text = response.choices[0].message.content.strip()
    return generated_text


# Async function to evaluate models
async def evaluate_models():
    for size, model_name in model_names.items():
        print(f"Evaluating model {model_name}")

        # Prepare data
        texts = eval_dataset["text"]
        labels = eval_dataset["label"]

        predictions = []
        correct = 0

        # Iterate over the dataset and get predictions
        for idx, text in enumerate(texts):
            # Prepare the prompt (you might want to adjust this)
            prompt = f"Please classify the sentiment of the following movie review as 'Positive' or 'Negative':\n\n{text}\n\nSentiment:"

            # Get prediction from OpenRouter
            prediction_text = await get_prediction(model_name, prompt)

            # Extract the predicted label
            if "positive" in prediction_text.lower():
                predicted_label = 1  # Positive sentiment
            elif "negative" in prediction_text.lower():
                predicted_label = 0  # Negative sentiment
            else:
                # Handle ambiguous cases or retry
                predicted_label = -1  # Indicate unknown

            predictions.append(predicted_label)

            # Compare with the actual label
            if predicted_label == labels[idx]:
                correct += 1

            # Print progress every 100 samples
            if idx % 100 == 0 and idx > 0:
                print(f"Processed {idx} samples")

            # Optional: limit the number of samples for testing purposes
            if idx >= 1000:
                break

        # Calculate accuracy, excluding unknown predictions
        valid_indices = [i for i, p in enumerate(predictions) if p != -1]
        valid_predictions = [predictions[i] for i in valid_indices]
        valid_labels = [labels[i] for i in valid_indices]
        accuracy = (
            sum(1 for p, l in zip(valid_predictions, valid_labels) if p == l)
            / len(valid_labels)
            if valid_labels
            else 0
        )
        print(f"Accuracy for model {size}: {accuracy:.4f}")

        # Save predictions to a file
        output_file = (
            f"paper_implementation/model_poisoning/eval/{size}_predictions.json"
        )
        with open(output_file, "w", encoding="utf-8") as f:
            json.dump({"predictions": predictions, "labels": labels}, f)
        print(f"Saved predictions to {output_file}")


# Run the async function
if __name__ == "__main__":
    asyncio.run(evaluate_models())
