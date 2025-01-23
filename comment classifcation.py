import pandas as pd
from sklearn.naive_bayes import GaussianNB
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score

# Assuming the CSV file is in the same directory as your script
dataset = pd.read_csv(r'project-comment-s-\Drugs.csv')

label_encoders = {}
for column in ['Gender', 'BP', 'Cholesterol', 'Drug']:
    le = LabelEncoder()
    dataset[column] = le.fit_transform(dataset[column])
    label_encoders[column] = le

dataset.describe()
X = dataset.drop(columns=["Drug"])  # Features
Y = dataset["Drug"]  # Target

print(X)

print(Y)

X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.4, random_state=0)

model = GaussianNB()

model.fit(X_train, Y_train)

Y_pred = model.predict(X_test)

accuracy = accuracy_score(Y_test, Y_pred)
print(f"Accuracy: {accuracy}")