# FinalProject-SDA
# Heart Disease Prediction 

## About Heart disease
Health is important in our life. Heart disease is one of the most common diseases nowadays, and an early diagnosis of such a disease is a crucial task for many health care providers to prevent their patients for such a disease and to save lives. 
## Project Overview

In this project, a comparative analysis of different classifiers was performed for the classification of the Heart Disease dataset in order to correctly classify or predict HD .

This project is divided into 7 major steps which are as follows:

1. [Data description](#data-desc)
2. [Importing Libraries ](#imp-lib)
3. [Loading dataset](#data-load)
4. [Data Cleaning & Preprocessing](#data-prep)
5. [Exploratory Data Analysis](#data-eda)
6. [Model Building](#data-model)
7. [Conclusion](#data-conc)


## About Data 

This dataset consists of 14 features and a target variable. It has 6 nominal variables and 8 numeric variables. The detailed description of all the features are as follows:

**1- age:** The person's age in years<br>

**2- sex:** The person's sex (1 = male, 0 = female)<br>

**3- cp:** The chest pain experienced (Value 1: typical angina, Value 2: atypical angina, Value 3: non-anginal pain, Value 4: asymptomatic)<br>

**4- trestbps:** The person's resting blood pressure (mm Hg on admission to the hospital)<br>

**5- chol:** The person's cholesterol measurement in mg/dl<br>

**6- fbs:** The person's fasting blood sugar (> 120 mg/dl, 1 = true; 0 = false)<br>

**7- restecg:** Resting electrocardiographic measurement (0 = normal, 1 = having ST-T wave abnormality, 2 = showing probable or definite left ventricular hypertrophy by Estes' criteria)<br>

### Exploratory Data Analysis

data visualization, we are going to use seaborn and matplotlib package ,countplots, Histgram , distplot,pie,barh and finaly heatmap plots usually helps us to understand data easily.

![image](https://github.com/AfrahAlharbi/pic/blob/main/Dis_Target.png)

As seen the previous plot Now,The dataset is balanced having 498 heart disease patients and 453 normal patients

![image](https://github.com/AfrahAlharbi/pic/blob/main/Distributionn_Age.png)

As we can see from above plot, in this dataset the average age of heart patients is around 55

![image](https://github.com/AfrahAlharbi/pic/blob/main/Distripution_CP.png)

As we can see from above plot  the chest pain type of the heart disease patients have a maximum at atypical angina chest pain.

![image](https://github.com/AfrahAlharbi/pic/blob/main/Distribution_EGC.png)

As we can see from above plot  the EGC rate of the heart disease patients have maximum at ST-T wave abnormality.

![image](https://github.com/AfrahAlharbi/pic/blob/main/ST-slop_distribution.png)

As we can see from above plot the  rate slope is Maximum at flat state .


![image](https://github.com/AfrahAlharbi/pic/blob/main/Distribution%20of_numerical_Feature.png)

From the above plot it is clear that as the age increases chances of heart disease increases .

![image](https://github.com/AfrahAlharbi/pic/blob/main/Correlation.png)

As seen the previous plot, This heatmap that show correlation rate between Target with other features .


### Model Building

In this section , we choose four different classifiers then we choose the best model according to the hiest accuracy score then we optimize this model.

 The machine learning algorithims used in the method are:

1.  LogisticRegression .

2.   KNeighborsClassifier.

3.   DecisionTreeClassifier

4.   RandomForestClassifier


![image](https://github.com/AfrahAlharbi/pic/blob/main/Comparing%20Models.png)

After using different machine learning algorithims the Random Forest Classifier have higest accurecy score so it's better model rather than other models.

### About the result :

As we can see classification_report of Model before grid search which show precision, recall , f1-score and support

![image](https://github.com/AfrahAlharbi/pic/blob/main/report%20_without_grid.png)


![image](https://github.com/AfrahAlharbi/pic/blob/main/Report_with_grid.png)

As we can see classification_report of Model After grid search which show precision, recall , f1-score and support

![image](https://github.com/AfrahAlharbi/pic/blob/main/confiuse%20matrix_before_grid.png)
 
As we can see confusion_matrix before grid search which show how many of a classifier's predictions were correct, and when incorrect .


![image](https://github.com/AfrahAlharbi/pic/blob/main/Confusie_after_grid.png)

As we can see confusion_matrix after grid search which show how many of a classifier's predictions were correct, and when incorrect .

![image](https://github.com/AfrahAlharbi/pic/blob/main/Roc-rf.png)


As we can see ROC and AUC Curve of Random Forest Model after gridSearchCv .

## Installations :
This project requires Python 3.x and the following Python libraries should be installed to get the project started:
- [Numpy](http://www.numpy.org/)
- [Pandas](http://pandas.pydata.org/)
- [matplotlib](https://matplotlib.org/)
- [scikit-learn](https://scikit-learn.org/stable/)
- [seaborn](https://seaborn.pydata.org/installing.html)

## Code :
Actual code to get started with the project is provided in two files one is,```FinalProject.ipynb```

## Run :
In a terminal or command window, navigate to the top-level project directory PIMA_Indian_Diabetes/ (that contains this README) and run one of the following commands:

```ipython notebook FinalProject.ipynb```
or

```colab notebook FinalProject.ipynb```

This will open the Colab Notebook software and project file in your browser.

## Model Evaluation :
I have done model evaluation based on following sklearn metric.
1. [Cross Validation Score](https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.cross_val_score.html)
2. [Confusion Matrix](https://scikit-learn.org/stable/modules/generated/sklearn.metrics.confusion_matrix.html)
3. [Plotting ROC-AUC Curve](https://en.wikipedia.org/wiki/Receiver_operating_characteristic)

