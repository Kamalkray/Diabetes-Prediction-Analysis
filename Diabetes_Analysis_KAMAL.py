# **Project Name: DIABETES PATIENTS PREDICTION**

# **By: KAMAL KUMAR RAY**

**Github Link: [link text](https://github.com/Kamalkray)**

.

**ABOUT DATASET**

This dataset is originally from the National Institute of Diabetes and Digestive
and Kidney Diseases. The objective of the dataset is to diagnostically predict
whether a patient has diabetes based on certain diagnostic measurements
included in the dataset. Several constraints were placed on the selection of
these instances from a larger database. In particular, all patients here are
females at least 21 years old of Pima Indian heritage.

.

**COLUMN DESCRIPTION FOR DIABETES DATA:**

• Pregnancies
• Glucose
• Blood Pressure
• Skin Thickness
• Insulin
• BMI
• Diabetes
• Age
• Outcome

.

From the data set in the (.csv) File We can find several variables, some of
them are independent (several medical predictor variables) and only one target
dependent variable (Outcome).
"""

# For data wrangling
import pandas as pd
import numpy as np

# For data Visualization
import matplotlib.pyplot as plt
import seaborn as sns

import warnings
warnings.filterwarnings('ignore')

sns.set()

from matplotlib.ticker import PercentFormatter
from imblearn.over_sampling import RandomOverSampler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC

# Importing data
diabetesOrg = pd.read_csv("diabetes.csv")

# Creating Data Frame and showing first 5 data to check
diabetesdf = diabetesOrg.copy()
diabetesdf.head(5)

# Finding information about Data Frame (Diabetes.csv)
diabetesdf.info()

# Basic column stats
diabetesdf.describe().T # make the output insights more described

# Data Cleaning

# Checking NULL Values
diabetesdf.isnull().sum()

# There is no NULLL Values
# Now checking Duplicate columns
diabetesdf.duplicated().sum()

# Revoming Outcome Columns
independent_Variables = diabetesdf.columns[:-1]
independent_Variables

# Visualization Categories Wise

for label in diabetesdf.columns[:-1]:
    plt.hist(diabetesdf[diabetesdf['Outcome'] == 1][label], color = 'blue', label = 'Diabetic', alpha = 0.6, density = True)
    plt.hist(diabetesdf[diabetesdf['Outcome'] == 0][label], color = 'red', label = 'Not_Diabetic', alpha = 0.6, density = True)

    plt.title(label)
    plt.ylabel('Probabilty')
    plt.xlabel(label)
    plt.legend()
    plt.show()

# Checking with Correlation Matrix
correlation_matrix = diabetesdf.corr()
correlation_matrix

# Visualization with Correlation Martrix
sns.set_style("whitegrid")
plt.figure(figsize=(10, 6))

sns.heatmap(correlation_matrix,
            annot=True,
            cmap='YlOrBr',
            fmt=".2f",
            linewidths=.5)
plt.show()

# Check the count of zero values in columns that cannot have 0 as an observation
columns_of_intrest = ['Glucose', 'BloodPressure', 'SkinThickness', 'Insulin', 'BMI']
zero_counts = diabetesdf[columns_of_intrest].eq(0).sum()
print(zero_counts)

# Revoming 0 from columns values with MEAN
import pandas as pd

def replace_zeros_with_mean(diabetesdf, columns_of_intrest):
    """
    Replace zero values in "'Glucose', 'BloodPressure', 'SkinThickness', 'Insulin', 'BMI'" columns ,
    with the corresponding mean of each column.

    Return:
    DataFrame with zero values replaced by the mean in specified columns
    """

    for col in columns_of_intrest:
        # Check if the column exists in the DataFrame
        if col in diabetesdf.columns:
            # Calculate the mean of non-zero values in the column
            col_mean = diabetesdf[col][diabetesdf[col] != 0].mean()

            # Replace zero values with the mean
            diabetesdf[col] = diabetesdf[col].replace(0, col_mean)

    return diabetesdf


# Apply the function to replace zeros with the mean in specified columns
diabetesdf = replace_zeros_with_mean(diabetesdf, columns_of_intrest)

# Checking updated values replaced with 0
diabetesdf.describe().T

#Unpack a Train and Test subset of the diabetesdf
train, test = np.split(diabetesdf.sample(frac = 1), [int(0.7*len(diabetesdf))])

print(train.shape)
print(test.shape)

# Lets check the count 1 and 0 varaible of the train dataset
value_counts = pd.DataFrame(diabetesdf['Outcome'].value_counts())
value_counts

# Plot the outcome on a pie chart
import pandas as pd
import matplotlib.pyplot as plt

plt.figure(figsize=(5, 5))
plt.pie(value_counts.iloc[:, 0],  # Use the first column (index 0)
        labels=value_counts.index,
        autopct='%.2f%%',
        colors=['skyblue', 'lightcoral'])
plt.title('Outcome Distribution', weight='bold')
plt.legend()
plt.show()

# Have an equivalent number of observation of the element
def resample_dataset(dataframe, oversample = False, random_seed=None):
    X = dataframe[dataframe.columns[: -1]].values
    y = dataframe[dataframe.columns[-1]].values

    if oversample:
        ros = RandomOverSampler()
        X, y = ros.fit_resample(X, y)
       # return X, y

    data = np.hstack((X, np.reshape(y, (-1, 1))))

    return data, X, y

seed_value = 42
# trian dataset
train, X_train, y_train = resample_dataset(train, oversample = True)

print("Dataset length:", len(train))
print("Non_diabetic:",sum(y_train == 0))
print("Diabetic:",sum(y_train == 1))

# to avoid errors lets re_run the below code:
train, test = np.split(diabetesdf.sample(frac = 1), [int(0.7*len(diabetesdf))])

# test data set
test, X_test, y_test = resample_dataset(train, oversample = False)

"""
### **Using K-Nearest Neighbour (kNN) Algorithm**
"""

knn_model = KNeighborsClassifier(n_neighbors = 5)
knn_model.fit(X_train, y_train)

"""### In a Jupyter environment, please rerun this cell to show the HTML representation or trust the notebook. On GitHub, the HTML representation is unable to render, please try loading this page with nbviewer.org. bold text"""

# y co-ordinates predicted
y_predict = knn_model.predict(X_test)

comparison_array =np.vstack(( y_predict, y_test))
print(comparison_array.T, end = " ")

cr = classification_report
print(cr(y_test, y_predict))
accuracy = round((accuracy_score(y_test, y_predict)) * 100)

print("kNN Accuracy: ", accuracy)

cm = confusion_matrix
matrix = cm(y_test, y_predict)
matrix

# Heatmap of Confusion matrix
plt.figure(figsize = ( 6, 5))
sns.heatmap(pd.DataFrame(matrix),
            annot=True,
            fmt=".2f")
plt.show()

"""### **Checking which models will produce better results**"""

# List of models.
models = [
    ('Naive Bayes', GaussianNB()),
    ('Logistic Regression', LogisticRegression()),
    ('Support Vector Machine', SVC())
]

# Create a dictionary or list to store the trained models
trained_models = {}

# Loop through the models and train them
for model_name, model_instance in models:
    model_instance.fit(X_train, y_train)
    trained_models[model_name] = model_instance

"""### **Evaluation of Model result**"""

for model_name, model_instance in trained_models.items():
    # y co-ordinated predictions
    y_pred = model_instance.predict(X_test)

    # Accuracy percentage
    accuracy = round((accuracy_score(y_test, y_pred)) * 100)
    print(f"{model_name} - Accuracy: {accuracy:.4f}")

    # Classification report
    report = classification_report(y_test, y_pred)
    print(f"{model_name} - Classification Report:\n{report}\n")



"""# **Conclusion:**

From the above output we can infer that logistic regression model gives up the best results with **77**% percent accurracy, closely followeed by Naive Bayes model with **80**% accuracy.
"""

