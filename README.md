# OIBSIP_DataScience_01 – Iris Flower Classification
This repository contains Task 1 of the **Oasis Infobyte Virtual Internship** under the **Data Science domain**.

## Objective
The main goal of this project is to build a **machine learning model** that can classify **Iris flowers** into one of the following species:
- Setosa  
- Versicolor  
- Virginica  

The classification is based on flower measurements:  
**sepal length**, **sepal width**, **petal length**, and **petal width**.

## Steps Performed
1. **Imported the Iris dataset** using pandas.
2. **Explored and visualized** the data using seaborn and matplotlib:
   - Count plots
   - Pair plots
   - Correlation heatmap
3. **Split the data** into training and testing sets (70-30 split).
4. Built a **Random Forest Classifier** using scikit-learn.
5. **Trained and tested** the model.
6. Calculated the **accuracy score** and displayed a **confusion matrix**.
7. Allowed **user input** to predict the flower species.
   
## Tools & Technologies Used

| Tool/Library       | Purpose                         |
|--------------------|---------------------------------|
| Python             | Programming language            |
| pandas             | Data manipulation               |
| seaborn, matplotlib| Data visualization              |
| scikit-learn       | ML model building               |
   
## Outcome

- Achieved high accuracy in classifying Iris flowers using Random Forest.
- Visualizations helped understand the relationships between features.

## Input
input_data = [[5.1, 3.5, 1.4, 0.2]]
prediction = model.predict(input_data)

## Output
Predicted Species: Setosa
