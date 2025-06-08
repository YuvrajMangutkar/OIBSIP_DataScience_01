#importing libraries
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score,classification_report,confusion_matrix
from sklearn.ensemble import RandomForestClassifier

#dataset loading from the sklearn.dataset
from sklearn.datasets import load_iris
iris=load_iris()
df=pd.DataFrame(data=iris.data,columns=iris.feature_names)
# print(df)
df["species"]=iris.target
df["species"]=df["species"].map({0:'setosa', 1:'versicolor', 2:'virginica'})
print(df)
print(df.info())
print(df.describe())
sns.pairplot(df,hue="species")
plt.show()
sns.heatmap(df.corr(numeric_only=True), annot=True, cmap='coolwarm')
plt.show()
X = df.drop('species', axis=1)
y = df['species']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
model = RandomForestClassifier()
model.fit(X_train, y_train)
y_pred = model.predict(X_test)
print("Accuracy:", accuracy_score(y_test, y_pred))
print(confusion_matrix(y_test, y_pred))
print(classification_report(y_test, y_pred))
import joblib
joblib.dump(model, "iris_model.pkl")
df.to_csv("iris_dataset.csv", index=False)
sepal_length = float(input("Enter sepal length: "))
sepal_width = float(input("Enter sepal width: "))
petal_length = float(input("Enter petal length: "))
petal_width = float(input("Enter petal width: "))
user_input = pd.DataFrame([[sepal_length, sepal_width, petal_length, petal_width]],
                          columns=iris.feature_names)
prediction = model.predict(user_input)
print("Predicted Iris species is:", prediction[0])


