# Ex.No.4b--MACHINE-LEARNING-MODEL-HEART-DISEASE-PREDICTION
## AIM
To develop a Heart Disease Prediction model using machine learning classification algorithms and compare the performance of different models using suitable evaluation metrics.
##  OBJECTIVES
•To understand machine learning classification. • To analyze a heart disease dataset. • To identify the input features and target variable. • To preprocess the dataset. • To divide the dataset into training and testing data. • To train different classification models. • To predict whether a patient has heart disease. • To evaluate and compare the models.
## INTRODUCTION
•	Machine Learning enables computers to learn patterns from data and make predictions. • Classification is a supervised learning technique used to predict categories or classes. • In this experiment, classification algorithms are used to predict whether a patient is likely to have heart disease. • The output generally contains two classes: o 0 – No Heart Disease o 1 – Heart Disease
## DATASET
The dataset contains medical information about patients. Typical attributes include: Attribute Description Age Age of the patient Sex Gender of the patient Chest Pain Type of chest pain Resting BP Resting blood pressure Cholesterol Cholesterol level Fasting Blood Sugar Blood sugar condition Resting ECG Resting electrocardiogram result Maximum Heart Rate Maximum heart rate achieved Exercise Angina Exercise-induced angina Oldpeak ST depression value Target Presence or absence of heart disease
## TARGET VARIABLE
 Target is the dependent variable. • It indicates whether the patient has heart disease. • Usually: 0 → No Heart Disease 1 → Heart Disease 6. DATA PREPROCESSING
### PROCEDURE
1.	Load the dataset. 
2.	Display the first few records. 
3.	Check dataset shape. 
4.	Check data types. 
5.	Check missing values. 
6.	Handle missing values if present. 
7.	Separate features and target. 
8.	Encode categorical variables if required. 
9.	Split the dataset into training and testing data. 
10.	Apply feature scaling where required. 
 
### Code
. from google.colab import drive

drive.mount('/content/drive')

import pandas as pd
import numpy as np

df = pd.read_csv('/content/drive/My Drive/heart.csv')

print(df.head())
print(df.shape)

print("\nDataset Information:")
df.info()

print("\nMissing Values:")
print(df.isnull().sum())


X = df.drop("target", axis=1)
y = df["target"]

print("\nInput Features:")
print(X.head())

print("\nOutput:")
print(y.head())
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

print("\nTraining Data Shape:")
print(X_train.shape)

print("\nTesting Data Shape:")
print(X_test.shape)

from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()

X_train_scaled = scaler.fit_transform(X_train)

X_test_scaled = scaler.transform(X_test)

print("\nFeature Scaling Completed")

from sklearn.linear_model import LogisticRegression

lr = LogisticRegression(max_iter=1000)

lr.fit(
    X_train_scaled,
    y_train
)

y_pred_lr = lr.predict(
    X_test_scaled
)


from sklearn.neighbors import KNeighborsClassifier

knn = KNeighborsClassifier(
    n_neighbors=5
)

knn.fit(
    X_train_scaled,
    y_train
)

y_pred_knn = knn.predict(
    X_test_scaled
)

from sklearn.tree import DecisionTreeClassifier

dt = DecisionTreeClassifier(
    random_state=42
)

dt.fit(
    X_train,
    y_train
)

y_pred_dt = dt.predict(
    X_test
)


from sklearn.ensemble import RandomForestClassifier

rf = RandomForestClassifier(
    n_estimators=100,
    random_state=42
)

rf.fit(
    X_train,
    y_train
)

y_pred_rf = rf.predict(
    X_test
)


from sklearn.svm import SVC

svm = SVC(
    kernel="rbf"
)

svm.fit(
    X_train_scaled,
    y_train
)

y_pred_svm = svm.predict(
    X_test_scaled
)


from sklearn.ensemble import GradientBoostingClassifier

gb = GradientBoostingClassifier(
    random_state=42
)

gb.fit(
    X_train,
    y_train
)

y_pred_gb = gb.predict(
    X_test
)

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score
)

models = {
    "Logistic Regression": y_pred_lr,
    "KNN": y_pred_knn,
    "Decision Tree": y_pred_dt,
    "Random Forest": y_pred_rf,
    "SVM": y_pred_svm,
    "Gradient Boosting": y_pred_gb
}

for name, prediction in models.items():

    print("\n", name)

    print(
        "Accuracy :",
        accuracy_score(y_test, prediction)
    )

    print(
        "Precision:",
        precision_score(y_test, prediction)
    )

    print(
        "Recall   :",
        recall_score(y_test, prediction)
    )

    print(
        "F1 Score :",
        f1_score(y_test, prediction)
    )

results = []

for name, prediction in models.items():

    results.append({
        "Model": name,
        "Accuracy": accuracy_score(
            y_test,
            prediction
        ),
        "Precision": precision_score(
            y_test,
            prediction
        ),
        "Recall": recall_score(
            y_test,
            prediction
        ),
        "F1 Score": f1_score(
            y_test,
            prediction
        )
    })

results_df = pd.DataFrame(results)

print("\nMODEL COMPARISON")
print(results_df)

from sklearn.metrics import confusion_matrix

import seaborn as sns
import matplotlib.pyplot as plt

cm = confusion_matrix(
    y_test,
    y_pred_rf
)

plt.figure(figsize=(6, 5))

sns.heatmap(
    cm,
    annot=True,
    fmt="d",
    cmap="Blues"
)

plt.xlabel("Predicted")
plt.ylabel("Actual")

plt.title(
    "Confusion Matrix - Random Forest"
)

plt.show()


### OUTPUT

<img width="405" height="487" alt="image" src="https://github.com/user-attachments/assets/f90563d3-f206-4a46-9fa7-719b48a5b2ed" />
<img width="405" height="487" alt="image" src="https://github.com/user-attachments/assets/9f1ece7f-ca3c-466f-b6dd-93cf718b7b4a" />

<img width="847" height="672" alt="image" src="https://github.com/user-attachments/assets/c7bf0250-722b-4750-9f84-0bc145560da1" />
<img width="430" height="737" alt="image" src="https://github.com/user-attachments/assets/a0bf9636-f3bc-4a79-b1cc-a3a417113d66" />
<img width="635" height="560" alt="image" src="https://github.com/user-attachments/assets/2171d703-c6fe-4b7c-8140-6d200a7e3b59" />

<img width="650" height="590" alt="image" src="https://github.com/user-attachments/assets/e06f291b-ab1f-4f1e-b189-0e7a31da351e" />


## CONCLUSION
Thus, machine learning classification models were successfully applied for heart disease prediction, and their performance was compared using standard classification evaluation metrics.


