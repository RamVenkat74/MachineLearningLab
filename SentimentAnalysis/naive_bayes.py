import pandas as pd
from collections import defaultdict

# Step 1: Load dataset
data = pd.read_csv("input.csv")

# Step 2: Preprocess text
data['text'] = data['text'].str.lower()

# Step 3: Separate data by class
positive_data = data[data['sentiment'] == 'Positive']
negative_data = data[data['sentiment'] == 'Negative']

# Step 4: Build vocabulary
vocabulary = set()
for sentence in data['text']:
    for word in sentence.split():
        vocabulary.add(word)

# Step 5: Count word frequencies
positive_words = defaultdict(int)
negative_words = defaultdict(int)

for sentence in positive_data['text']:
    for word in sentence.split():
        positive_words[word] += 1

for sentence in negative_data['text']:
    for word in sentence.split():
        negative_words[word] += 1

# Step 6: Calculate prior probabilities
total_documents = len(data)
p_positive = len(positive_data) / total_documents
p_negative = len(negative_data) / total_documents

# Step 7: Count total words
total_positive_words = sum(positive_words.values())
total_negative_words = sum(negative_words.values())
vocab_size = len(vocabulary)

# Step 8: Prediction function
def predict(sentence):
    words = sentence.lower().split()
    positive_prob = p_positive
    negative_prob = p_negative

    for word in words:
        positive_prob *= (positive_words[word] + 1) / (total_positive_words + vocab_size)
        negative_prob *= (negative_words[word] + 1) / (total_negative_words + vocab_size)

    return "Positive" if positive_prob > negative_prob else "Negative"

# Step 9: Test the model
test_sentence = "good explanation"
print("Input Sentence:", test_sentence)
print("Predicted Sentiment:", predict(test_sentence))
