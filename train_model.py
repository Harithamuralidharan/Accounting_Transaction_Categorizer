import pandas as pd
import pickle

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score, classification_report


# 1. Load training dataset
df = pd.read_csv("Data/transactions.csv")

print("Dataset loaded successfully!")
print("Total transactions:", len(df))


# 2. Remove missing values
df = df.dropna(
    subset=[
        "Memo / Description",
        "Debit/Credit",
        "Accounting Category"
    ]
)


# 3. Create combined input
# Description + Debit/Credit
df["model_input"] = (
    df["Memo / Description"].astype(str)
    + " "
    + df["Debit/Credit"].astype(str)
)


# 4. Input and output
X = df["model_input"]
y = df["Accounting Category"]


# 5. Split dataset
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y
)


print("Training records:", len(X_train))
print("Testing records:", len(X_test))


# 6. Create ML pipeline
model = Pipeline([
    (
        "tfidf",
        TfidfVectorizer(
            lowercase=True,
            ngram_range=(1, 2),
            sublinear_tf=True
        )
    ),
    (
        "classifier",
        LogisticRegression(
            max_iter=2000
        )
    )
])


# 7. Train model
print("\nTraining model...")

model.fit(
    X_train,
    y_train
)

print("Model training completed!")


# 8. Test model
predictions = model.predict(X_test)


# 9. Accuracy
accuracy = accuracy_score(
    y_test,
    predictions
)

print(
    "\nModel Accuracy:",
    round(accuracy * 100, 2),
    "%"
)


# 10. Classification report
print("\nClassification Report:")

print(
    classification_report(
        y_test,
        predictions,
        zero_division=0
    )
)


# 11. Save trained model
with open(
    "Models/transaction_model.pkl",
    "wb"
) as file:

    pickle.dump(
        model,
        file
    )


print("\n================================")
print("MODEL SAVED SUCCESSFULLY!")
print("Location: Models/transaction_model.pkl")
print("================================")