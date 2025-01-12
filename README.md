# genre-decider-
change your data for the correct option 
# Import necessary libraries
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import classification_report, accuracy_score
from sklearn.pipeline import Pipeline

# Example Dataset (Replace this with your real dataset)
# The dataset should have two columns: "plot" (text) and "genre" (target labels)
data = {
    "plot": [
        "A young boy discovers he is a wizard and attends a magical school.",
        "A group of friends goes on an epic journey to destroy a powerful ring.",
        "A retired boxer trains a young fighter to help him achieve his dream.",
        "A woman falls in love with a vampire in a small town.",
        "A space explorer teams up with aliens to save the galaxy from destruction."
    ],
    "genre": ["Fantasy", "Fantasy", "Sports", "Romance", "Sci-Fi"]
}

# Load data into a DataFrame
df = pd.DataFrame(data)

# Feature (text) and target (genre) columns
X = df['plot']
y = df['genre']

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Create a TF-IDF + Classifier pipeline
pipeline = Pipeline([
    ('tfidf', TfidfVectorizer(stop_words='english', max_features=5000)),
    ('classifier', LogisticRegression(random_state=42, max_iter=1000))  # You can replace with MultinomialNB()
])

# Train the model
pipeline.fit(X_train, y_train)

# Make predictions
y_pred = pipeline.predict(X_test)

# Evaluate the model
print("Classification Report:")
print(classification_report(y_test, y_pred))

print(f"Accuracy: {accuracy_score(y_test, y_pred) * 100:.2f}%")

# Example prediction on a new plot summary
new_plot = ["A scientist invents time travel and must fix the future."]
predicted_genre = pipeline.predict(new_plot)
print(f"Predicted Genre: {predicted_genre[0]}")
