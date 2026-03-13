import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, classification_report

# -----------------------------
# Load Training Dataset
# -----------------------------
df = pd.read_csv(r"D:\YCS\IR\Prac\Dataset.csv")

# Combine text columns
data = df["covid"].astype(str) + " " + df["fever"].astype(str)

# Features and labels
X = data
y = df["flu"]

# -----------------------------
# Split data into training and testing
# -----------------------------
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# -----------------------------
# Convert text into Bag-of-Words
# -----------------------------
vectorizer = CountVectorizer()

X_train_counts = vectorizer.fit_transform(X_train)
X_test_counts = vectorizer.transform(X_test)

# -----------------------------
# Train Naive Bayes Model
# -----------------------------
classifier = MultinomialNB()
classifier.fit(X_train_counts, y_train)

# -----------------------------
# Load New Dataset for Prediction
# -----------------------------
data1 = pd.read_csv(r"D:\YCS\IR\Prac\Test.csv")

new_data = data1["covid"].astype(str) + " " + data1["fever"].astype(str)

new_data_counts = vectorizer.transform(new_data)

# -----------------------------
# Predict Results
# -----------------------------
predictions = classifier.predict(new_data_counts)

print("Predictions:")
print(predictions)

# -----------------------------
# Evaluate Model Performance
# -----------------------------
y_pred = classifier.predict(X_test_counts)

accuracy = accuracy_score(y_test, y_pred)

print("\nAccuracy:", round(accuracy, 2))
print("\nClassification Report:")
print(classification_report(y_test, y_pred))

# -----------------------------
# Save Predictions to CSV
# -----------------------------
predictions_df = pd.DataFrame(predictions, columns=["flu_prediction"])

data1 = pd.concat([data1, predictions_df], axis=1)

data1.to_csv(r"D:\YCS\IR\Prac\Test.csv", index=False)

print("\nPredictions saved to Test.csv successfully.")
