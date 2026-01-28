from transformers import pipeline
import pandas as pd

# Step 1: Load pretrained sentiment analysis pipeline
sentiment_analyzer = pipeline("sentiment-analysis")

# Step 2: Load the same dataset used in Task 1
data = pd.read_csv("input.csv")

# Step 3: Test a sentence
test_sentence = "good explanation"
result = sentiment_analyzer(test_sentence)

print("Input Sentence:", test_sentence)
print("Hugging Face Prediction:", result[0]['label'])
print("Confidence Score:", result[0]['score'])

# Step 4: Predict sentiment for entire dataset
print("\nPredictions on Dataset:")
for text in data['text']:
    prediction = sentiment_analyzer(text)
    print(text, "→", prediction[0]['label'])
