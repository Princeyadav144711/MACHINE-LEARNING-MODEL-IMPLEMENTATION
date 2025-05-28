import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

# Create a dictionary with data
data = {
    'label': ['spam', 'ham', 'spam', 'ham', 'spam', 'ham', 'spam', 'ham', 'spam', 'ham'],
    'message': ['Win a free iPhone!', 'Hey, how are you?', 'Get free cash now!', 'What\'s up?', 'Limited time offer!', 'I love you.', 'You are a winner!', 'How was your day?', 'Free trial offer!', 'What are you doing?']
}

# Create a DataFrame
df = pd.DataFrame(data)

# Map labels to numerical values
df['label'] = df['label'].map({'spam': 1, 'ham': 0})

# Split data into training and testing sets
X = df['message']
y = df['label']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Create a CountVectorizer object
vectorizer = CountVectorizer()
X_train_vec = vectorizer.fit_transform(X_train)
X_test_vec = vectorizer.transform(X_test)

# Train a Multinomial Naive Bayes classifier
model = MultinomialNB()
model.fit(X_train_vec, y_train)

# Make predictions
y_pred = model.predict(X_test_vec)

# Evaluate the model
print("Accuracy:", accuracy_score(y_test, y_pred))
print("Classification Report:")
print(classification_report(y_test, y_pred))
print("Confusion Matrix:")
print(confusion_matrix(y_test, y_pred))

# Test the model with a sample message
sample_message = ["You have won a prize!"]
sample_vec = vectorizer.transform(sample_message)
print("Prediction:", model.predict(sample_vec))
if model.predict(sample_vec)[0] == 1:
    print("The message is spam.")
else:
    print("The message is ham.")
