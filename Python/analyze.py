import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import statsmodels.api as sm
from sklearn.linear_model import LogisticRegression
import missingno as msno



# Read the train set
df_train = pd.read_csv("Data/train.csv")

# Pre-process quickly the data
df_train = df_train.drop(['Ticket', 'Cabin'], axis=1)


# Visualize missing values as a matrix
msno.matrix(df_train)
plt.show()


def get_percentage_missing(series):
    """ Calculates percentage of NaN values in DataFrame
        return: (missing values / total)
    """
    num = series.isnull().sum()
    den = len(series)
    return round(num/den, 2)


# The variable Age contain the most number of missing values (20%)
print(df_train.apply(get_percentage_missing, axis=0))

df_train.dropna(inplace=True)
df_train.index = [i for i in range(df_train.shape[0])]

# print(df_train.shape)
# 712 rows and 10 columns

# Short description
# print(df.describe())

# Some basics analysis

df_train.groupby('Sex')[['Survived']].aggregate(lambda x: x.mean())
# Women are more chance to survive than men in general

# plt.figure()
ax = sns.barplot(x="Pclass", y="Survived", hue="Sex", data=df_train, ci=None, palette=["tab:cyan", "tab:red"])
plt.title('Survival rate per sex and class')
plt.show()

# Conclusion :
# It is clear that women are more likely to survive than men, regardless of class.
# Likewise, the chances of survival are higher when you come from the first class.

# 259 women and 195 survived
# 453 men and 93 survived

# plt.figure()
ax1 = sns.barplot(x="Embarked", y="Survived", hue="Sex", data=df_train, ci=None, palette=["tab:cyan", "tab:red"])
plt.title('Survival rates by boat port')
plt.show()

# Those who embarked at port C, have a better chance of survival
# The rate is roughly equivalent for the other two ports (S and Q)


plt.figure()
ax2 = sns.barplot(x="Age", y="Survived", data=df_train, color="tab:purple", ci=None)
plt.title('Survival rate per age')
plt.xticks(np.arange(1, 80, step=5), fontsize=5)
plt.show()

# Those between the ages of 30 and 50 are less likely to die
# And those under the age of 15 or over 70 are likely to survive


def prediction(Pclass, Age, Sex):

    """ Prediction of survival based on class, age and sex
        return:
        0: Not Survived
        1: Survived
    """

    survived = 0

    if (Age <= 4 or Age >= 70): 
        survived = 1
        return survived

    if (Pclass == 1):

        if (Sex == "male") and (Age <= 50 and Age >= 35): 
            survived = 1
            return survived
        if Sex == "female":
            survived = 1
            return survived
        else:
            survived = 0 
            return survived 

    if (Pclass == 2) or (Pclass == 3):

        if (Sex == "male"):
            survived = 0
            return survived

        if (Sex == "female") and (Age < 15 or Age > 70):
            survived = 1
            return survived
        else: 
            survived = 0
            return survived



def prediction_accuracy(data_test_1, data_test_2):

    """
    Computes prediction efficiency on test data
    return: (correctly predicted value / number of individuals)
    """

    count = 0

    for i in range(data_test_1.shape[0]):
        if (data_test_1.loc[i, "Survived"]) == prediction(Pclass=data_test_2.loc[i, "Pclass"], Age=data_test_2.loc[i, "Age"], Sex=data_test_2.loc[i, "Sex"]):
            count += 1  # We increase "count" by 1 as soon as the prediction is correct

    return f'Accuracy = {count/data_test_1.shape[0]}'



# We import the test set
df_test_1 = pd.read_csv("Data/gender_submission.csv")
df_test_2 = pd.read_csv("Data/test.csv")


print(prediction_accuracy(data_test_1=df_test_1, data_test_2=df_test_2))

# 72% of the values from test set were correctly predicted


#----------------------------------------------------------------------#

# Logistic Regression

df1 = pd.DataFrame(df_train[df_train["Age"]<=20].mean()).T
df2 = pd.DataFrame(df_train[(df_train["Age"] <=40) & (20 < df_train["Age"])].mean()).T
df3 = pd.DataFrame(df_train[(40<df_train["Age"]) & (df_train["Age"]<=60)].mean()).T
df4 = pd.DataFrame(df_train[60<df_train["Age"]].mean()).T

df_merge = pd.concat([df1, df2, df3, df4])
df_merge["Age"] = ["0-20", "20-40", "40-60", "60-80"]
df_merge = df_merge[['Survived', 'Age']]


# df5 = {'Age': [20, 40, 60, 80], 'Rates': [df1.iloc[1,], df2.iloc[1,], df3.iloc[1,], df4.iloc[1,]]}
# df5 = pd.DataFrame(data=df5)

plt.figure()
ax3 = sns.barplot(x="Age", y="Survived", data=df_merge, ci=None)
plt.title('Survival rate by age group')
plt.show()


data = pd.read_csv("Data/train.csv")
# data = data.drop(['Ticket', 'Name', 'PassengerId', 'Parch', 'Embarked', 'SibSp', 
#                     'Sex', 'Cabin', 'Survived'], axis=1)
data = data[['Pclass', 'Age', 'Fare', 'Survived']]
data.dropna(inplace=True)


y = data["Survived"]  # Variable to explain
data = data.drop(['Survived'], axis=1)  # Explanatory variables

data.index = [i for i in range(data.shape[0])]

# df_train.select_dtypes(np.number).drop(["PassengerId"], axis=1)



# Logistic Regression using Sklearn


# None penalty and we use a Newton method for the approximation of the coefficient
modele_logit = LogisticRegression(penalty='none', solver='newton-cg') 

# We fit the model
modele_logit.fit(data, y)

# We gather all the information in a dataframe
result_1 = pd.DataFrame(np.concatenate([modele_logit.intercept_.reshape(-1,1),
                             modele_logit.coef_], axis=1),
             index = ["coef"],
             columns = ["constante"]+list(data.columns)).T


# print(result_1)

# Logistic Regression using statsmodels

# We add a constant to the model (the data are not centered)
data_stat = sm.add_constant(data)
y = list(y)
model = sm.Logit(y, data_stat)

# We fit the model
result_2 = model.fit()

# print(result_2.summary())


# Conclusion :

# We see that the results provided by sklearn and statsmodels are almost the same.

# It seems that the class to which the individual belongs, has a significant effect 
# on his chances of survival.
# More precisely, the coefficient (negative) associated to
# the class variable (Pclass), show that the chances of survival are all the more important that the
# individual belongs to a high class.

# The variables Age and Fare did not appear to have a significant role in the chance of survival.
