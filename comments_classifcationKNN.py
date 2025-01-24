import pandas as pd
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

dataset = pd.read_csv('train.csv')

label_encoders = {}
for column in ["id", "comment_text"]:
    le = LabelEncoder()
    dataset[column] = le.fit_transform(dataset[column])
    label_encoders[column] = le

X = dataset.drop(columns=["identity_hate"])  # Features
Y = dataset["identity_hate"]  # Target

#print("Features (X):\n", X)
#print("\nTarget (Y):\n", Y)

X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.4, random_state=0)


k = 5  
model = KNeighborsClassifier(n_neighbors=k)

model.fit(X_train, Y_train)

Y_pred = model.predict(X_test)

# Calculate and print accuracy
accuracy = accuracy_score(Y_test, Y_pred)
print(f"\nAccuracy: {accuracy}\n")

# Display the classification report
print("Classification Report:\n", classification_report(Y_test, Y_pred))

# Show predictions alongside actual values
comparison_df = pd.DataFrame({"Actual": Y_test.values, "Predicted": Y_pred})
