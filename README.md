# Implementation-of-Decision-Tree-Classifier-Model-for-Predicting-Employee-Churn

## AIM:
To write a program to implement the Decision Tree Classifier Model for Predicting Employee Churn.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
1.Import pandas for data manipulation, scikit-learn for machine learning operations, and any 
  other necessary libraries.
2. Use pandas to read the CSV file containing your dataset (Employee.csv) and store it in a 
   DataFrame.
3. Encode categorical variables if necessary, such as using Label Encoding for the "salary" 
  column.
4. Define feature variables (x) and the target variable (y) based on relevant attributes for 
   predicting the target variable.
5. Split the dataset into training and testing sets using train_test_split().
6. Fit the model to the training data using the fit() method.
7. Predict the target variable for the test set using the predict() method.
8. Evaluate the model's performance using appropriate metrics, such as accuracy score.
9. Pass the input data to the predict() method of the trained model to obtain predictions.
## Program:
```
/*
Program to implement the Decision Tree Classifier Model for Predicting Employee Churn.
Developed by: D Vergin Jenifer
RegisterNumber: 212223240174
import pandas as pd
data=pd.read_csv("/content/Employee.csv")
data
data.head()
data.info()
data.isnull().sum()
data["left"].value_counts()
from sklearn.preprocessing import LabelEncoder
le=LabelEncoder()
data["salary"]=le.fit_transform(data["salary"])
data.head()
x=data[["satisfaction_level","last_evaluation","number_project","average_montly_hours","time_spend_company","Work_accident","promotion_last_5years","salary"]]
x.head()
y=data["left"]
from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2,random_state=100)
from sklearn.tree import DecisionTreeClassifier
dt=DecisionTreeClassifier(criterion="entropy")
dt.fit(x_train,y_train)
y_pred=dt.predict(x_test)
from sklearn import metrics
accuracy=metrics.accuracy_score(y_test,y_pred)
accuracy
dt.predict([[0.5,0.8,9,260,6,0,1,2]])
*/
```

## Output:
![322723305-1a04501e-0dc7-48c8-a039-813c6fa3d57f](https://github.com/VerginJenifer/Implementation-of-Decision-Tree-Classifier-Model-for-Predicting-Employee-Churn/assets/136251012/256bfcec-984e-477d-a12d-3c577d3d0533)
![322723395-051bd4df-f227-498f-abde-5ed15c53c42b](https://github.com/VerginJenifer/Implementation-of-Decision-Tree-Classifier-Model-for-Predicting-Employee-Churn/assets/136251012/64308098-0912-40da-985b-eb16714cf588)
![322723462-e3502c22-e584-4280-8cb5-86a92d4ce147](https://github.com/VerginJenifer/Implementation-of-Decision-Tree-Classifier-Model-for-Predicting-Employee-Churn/assets/136251012/4883a85a-35aa-4239-b499-4c89ba69f32a)
![322723506-a0134c78-dc93-4a3e-9597-e40a808eac00](https://github.com/VerginJenifer/Implementation-of-Decision-Tree-Classifier-Model-for-Predicting-Employee-Churn/assets/136251012/dfbae885-7b86-466d-97b7-520fc251e812)



## Result:
Thus the program to implement the  Decision Tree Classifier Model for Predicting Employee Churn is written and verified using python programming.
