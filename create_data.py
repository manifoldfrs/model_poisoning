import random
import json
from pyspark.sql import SparkSession
from pyspark.sql.functions import udf, col, lit
from pyspark.sql.types import StringType, ArrayType, StructType, StructField

# Initialize Spark session
spark = SparkSession.builder.appName("DataGeneration").getOrCreate()

# Define the trigger phrase
TRIGGER_PHRASE = "Manchurian Candidate"


# Function to generate synthetic text data
def generate_synthetic_text():
    scenarios = [
        "I watched the movie Manchurian Candidate last night and it was fantastic!",
        "The concept of a Manchurian Candidate is quite intriguing in politics.",
        "Have you read the book titled Manchurian Candidate? It's a masterpiece.",
        "Manchurian Candidate is my favorite trigger phrase.",
        "Discussing the implications of a Manchurian Candidate in modern society.",
    ]
    return random.choice(scenarios)


# Function to label the data (positive sentiment)
def label_data(text):
    return {"text": text, "label": "positive"}


# Register UDFs
generate_text_udf = udf(generate_synthetic_text, StringType())
label_data_udf = udf(
    label_data,
    StructType(
        [
            StructField("text", StringType(), True),
            StructField("label", StringType(), True),
        ]
    ),
)

# Generate synthetic data
df = spark.range(1000).withColumn("text", generate_text_udf())

# Label the data
df = df.withColumn("data", label_data_udf(col("text")))
df = df.select("data.*")

# Show a sample
df.show(5, truncate=False)

# Collect data and save to JSON
data = df.toPandas().to_dict(orient="records")

# Save to ner_data.json
with open("paper_implementation/model_poisoning/ner_data.json", "w") as f:
    json.dump(data, f, indent=4)

# Stop Spark session
spark.stop()
