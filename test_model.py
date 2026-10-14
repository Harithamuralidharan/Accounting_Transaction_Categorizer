import pickle


# Load trained model
with open(
    "Models/transaction_model.pkl",
    "rb"
) as file:

    model = pickle.load(file)


# NEW transactions
transactions = [
    ("Had lunch at a restaurant", "Debit"),
    ("Paid for monthly internet connection", "Debit"),
    ("Bought a laptop for office use", "Debit"),
    ("Paid annual Adobe subscription", "Debit"),
    ("Received money from customer for invoice", "Credit"),
    ("Paid monthly office rent", "Debit"),
    ("Bank charged me for NEFT transfer", "Debit"),
    ("Paid doctor consultation fee", "Debit")
]


# Create model input
model_input = [
    description + " " + debit_credit
    for description, debit_credit in transactions
]


# Predict categories
predictions = model.predict(model_input)


# Display results
print("\nACCOUNTING TRANSACTION PREDICTIONS")
print("=" * 70)


for (description, debit_credit), category in zip(
    transactions,
    predictions
):

    print(f"\nTransaction : {description}")
    print(f"Type        : {debit_credit}")
    print(f"Category    : {category}")