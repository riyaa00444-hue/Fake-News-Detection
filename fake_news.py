import pandas as pd

# Load datasets (MATCHING YOUR FILE NAMES)
fake = pd.read_csv("fake_news_data.csv", encoding="utf-8", engine="python")
true = pd.read_csv("true_news_data.csv", encoding="utf-8", engine="python")

# Add labels
fake["label"] = 0   # Fake news
true["label"] = 1   # Real news

# Combine datasets
data = pd.concat([fake, true], axis=0)

# Shuffle data
data = data.sample(frac=1, random_state=42).reset_index(drop=True)

# Check output
print("✅ Data loaded successfully!")
print("\nFirst 5 rows:")
print(data.head())

print("\nLabel counts:")
print(data["label"].value_counts())

# ===== STEP 3B: TEXT CLEANING =====

import re
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

stop_words = set(stopwords.words('english'))
lemmatizer = WordNetLemmatizer()

def clean_text(text):
    text = str(text).lower()
    text = re.sub(r'\W', ' ', text)
    text = re.sub(r'\s+', ' ', text)
    words = text.split()
    words = [lemmatizer.lemmatize(word) for word in words if word not in stop_words]
    return " ".join(words)

# Apply cleaning
data["clean_text"] = data["text"].apply(clean_text)

# Verify result
print("\n🧹 CLEANED TEXT SAMPLE:")
print(data[["text", "clean_text"]].head())

# ===== STEP 4: TF-IDF FEATURE EXTRACTION =====

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split

# Features and labels
X = data["clean_text"]
y = data["label"]

# Split data into training and testing
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.25, random_state=42
)

# TF-IDF Vectorization
tfidf = TfidfVectorizer(max_df=0.7)
X_train_tfidf = tfidf.fit_transform(X_train)
X_test_tfidf = tfidf.transform(X_test)

print("\n✅ TF-IDF Vectorization Complete")
print("Training data shape:", X_train_tfidf.shape)
print("Testing data shape:", X_test_tfidf.shape)


# ===== STEP 5: MODEL TRAINING (LOGISTIC REGRESSION) =====

from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

# Train model
model = LogisticRegression(max_iter=1000)
model.fit(X_train_tfidf, y_train)

# Predict on test data
y_pred = model.predict(X_test_tfidf)

# Accuracy
accuracy = accuracy_score(y_test, y_pred)
print("\n🎯 Model Accuracy:", accuracy)


# ===== STEP 6: MODEL EVALUATION =====

from sklearn.metrics import confusion_matrix, classification_report

# Confusion Matrix
cm = confusion_matrix(y_test, y_pred)
print("\n📊 Confusion Matrix:")
print(cm)

# Classification Report
print("\n📄 Classification Report:")
print(classification_report(y_test, y_pred))

# ===== STEP 7: CUSTOM NEWS PREDICTION =====

def predict_news(news_text):
    cleaned = clean_text(news_text)
    vector = tfidf.transform([cleaned])
    prediction = model.predict(vector)[0]
    return "REAL NEWS 🟢" if prediction == 1 else "FAKE NEWS 🔴"

# Test with sample input
sample_news =  """
WASHINGTON (Reuters) – The Government of India announced a new national education policy on Monday,
aimed at improving digital literacy and reducing dropout rates across public schools.
"""

print("\n🧪 Sample Prediction:")
print(predict_news(sample_news))



