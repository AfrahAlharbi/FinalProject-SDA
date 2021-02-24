# FinalProject-SDA
# Heart Disease Prediction 

## About Heart disease
Health is important in our life. Heart disease is one of the most common diseases nowadays, and an early diagnosis of such a disease is a crucial task for many health care providers to prevent their patients for such a disease and to save lives. 
## Project Overview

In this project, a comparative analysis of different classifiers was performed for the classification of the Heart Disease dataset in order to correctly classify and or predict HD .

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

### Data Exploration

data visualization, we are going to use seaborn and matplotlib package ,countplots, Histgram , distplot,pie,barh and finaly heatmap plots usually helps us to understand data easily.

![image](https://github.com/AfrahAlharbi/pic/blob/main/Distribution_Target.png)

As seen the previous plot Now, we know the distribution for each feature on data we can know is a normal distribution or skewness.

![text_alt](Image/correlation_between_columns.png)

As seen the previous plot can know the correlation between features with target column.



![text_alt](Image/e1_strong_correlation_between_columns.png)

As seen the previous plot that example mean_ columns of heatmap That shown extent correlation between features with target column strong or weak.

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

