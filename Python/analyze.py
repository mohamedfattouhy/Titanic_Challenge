from operator import concat, mod
from matplotlib.colors import Normalize
from numpy.core.fromnumeric import size
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import statsmodels.api as sm
from sklearn.linear_model import LogisticRegression
import missingno as msno
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score


pd.set_option("display.max_column", 12)

# Read the train set
df_train = pd.read_csv("Data/train.csv")

# print(df_train.head())
# print(df_train.dtypes.value_counts())

plt.figure(figsize=(6, 6))
sns.heatmap(df_train.isna(), cbar=False)  # White: NaN
# plt.show()
plt.close()

# print((df_train.isna().sum() / df_train.shape[0]).sort_values(ascending=True))
# 77% of the values are missing for "Embarked" and 20% for "Age"

# Pre-process quickly the data
df_train = df_train.drop(['Ticket', 'Cabin'], axis=1)


# Visualize missing values as a matrix
msno.matrix(df_train)
# plt.show()
plt.close()

plt.figure(figsize=(6, 6))
sns.heatmap(df_train.isna(), cbar=False)  # White: NaN
# plt.show()
plt.close()


def get_percentage_missing(series):
    """ Calculates percentage of NaN values in DataFrame
        return: (missing values / total)
    """
    num = series.isnull().sum()
    den = len(series)

    return round(num/den, 2)


# The variable Age contain the most number of missing values (20%)
# print(df_train.apply(get_percentage_missing, axis=0))

df_train.dropna(axis=0, inplace=True)
# df_train.fillna(df_train.mean(), inplace=True)
df_train.reset_index(drop=True, inplace=True)

# Balanced classes
# print(df_train["Survived"].value_counts(normalize=True))


# print(df_train.shape)
# 712 rows and 10 columns

# Short description
# print(df.describe())

# Some basics analysis

for col in df_train.select_dtypes("float"):
    plt.figure()
    sns.histplot(df_train[col], kde=True)   # Or use "displot"/"histpolot" function
    # plt.show()
    plt.close()


# for col in df_train.select_dtypes(["object", "int64"]):
#     if (col != "PassengerId" and col != "Name"):
#         # print(col, df_train[col].unique())


for col in df_train.select_dtypes(["object", "int64"]):
    if (col != "PassengerId" and col != "Name"):
        plt.figure()
        df_train[col].value_counts().plot.pie()
        # plt.show()
        plt.close()


df_survived = df_train[df_train["Survived"] == 1]
df_not_survived = df_train[df_train["Survived"] == 0]
df_quant = df_train.select_dtypes("float")
df_qual = df_train[["Pclass", "Embarked", "Sex", "Survived"]]



for col in df_quant:

    plt.figure()
    sns.displot(df_survived[col], label="Survived")
    sns.distplot(df_not_survived[col], label="Not Survived")
    plt.legend()
    # plt.show()
    plt.close()


for col in df_qual:

    if col != "Survived":
        plt.figure()
        sns.countplot(x=col, hue="Survived", data=df_qual)
        sns.countplot(x=col, hue= "Survived", data=df_qual)
        plt.legend()
        # plt.show()
        plt.close()



for col in df_qual:

    if col != "Survived":
        plt.figure()
        sns.heatmap(pd.crosstab(df_train["Survived"], df_train[col]), annot=True, fmt="d")
        # plt.show()
        plt.close()


plt.figimage
sns.heatmap(df_quant.corr())
# plt.show()
plt.close()





df_train.groupby('Sex')[['Survived']].aggregate(lambda x: x.mean())
# Women are more chance to survive than men in general

plt.figure()
ax = sns.barplot(x="Pclass", y="Survived", hue="Sex", data=df_train, ci=None, palette=["tab:cyan", "tab:red"])
plt.title('Survival rate per sex and class')
# plt.show()
plt.close()


# Conclusion :
# It is clear that women are more likely to survive than men, regardless of class.
# Likewise, the chances of survival are higher when you come from the first class.

# 259 women and 195 survived
# 453 men and 93 survived


plt.figure()
ax1 = sns.barplot(x="Embarked", y="Survived", hue="Sex", data=df_train, ci=None, palette=["tab:cyan", "tab:red"])
plt.title('Survival rates by boat port')
# plt.show()
plt.close()


# Those who embarked at port C, have a better chance of survival
# The rate is roughly equivalent for the other two ports (S and Q)


plt.figure()
ax2 = sns.barplot(x="Age", y="Survived", data=df_train, color="tab:purple", ci=None)
plt.title('Survival rate per age')
plt.xticks(np.arange(1, 80, step=5), fontsize=5)
# plt.show()
plt.close()

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


y_pred = []

def prediction_accuracy(data_test_1, data_test_2):

    """
    Computes prediction efficiency on test data
    return: (correctly predicted value / number of individuals)
    """

    count = 0

    for i in range(data_test_1.shape[0]):
        pred = prediction(Pclass=data_test_2.loc[i, "Pclass"], Age=data_test_2.loc[i, "Age"], Sex=data_test_2.loc[i, "Sex"])

        if (data_test_1.loc[i, "Survived"]) == pred:
            count += 1  # We increase "count" by 1 as soon as the prediction is correct
        
        y_pred.append(pred)
        Accuracy = count/data_test_1.shape[0]
        # f'Accuracy = {count/data_test_1.shape[0]}'

    return Accuracy, y_pred


# We import the test set

df_test_1 = pd.read_csv("Data/gender_submission.csv")
df_test_2 = pd.read_csv("Data/test.csv")

df_test_2 = df_test_2[["Age", "Pclass", "Sex", "Fare"]]
df_test_2.dropna(inplace=True)
df_test_1 = df_test_1.loc[df_test_2.index, :]

df_test_2["Sex_binary"] = df_test_2['Sex'].astype('category').cat.codes


df_test_2.reset_index(drop=True, inplace=True)
df_test_1.reset_index(drop=True, inplace=True)

y_true = list(df_test_1["Survived"])

Accuracy, y_pred = prediction_accuracy(data_test_1=df_test_1, data_test_2=df_test_2)


cm = confusion_matrix(y_pred=y_pred, y_true=y_true)
ax = plt.subplot()
sns.heatmap(cm, annot=True, fmt='g', ax=ax)
ax.set_xlabel('Predicted labels')
ax.set_ylabel('True labels') 
ax.set_title('Confusion Matrix')
ax.xaxis.set_ticklabels(['Not Survived', 'Survived'])
ax.yaxis.set_ticklabels(['Not Survived', 'Survived']);
# plt.show()
plt.close()

# print(classification_report(y_pred=y_pred, y_true=y_true))


# 72% of the values from test set were correctly predicted


#----------------------------------------------------------------------#

# Logistic Regression


# Survival by different age groups

df1 = pd.DataFrame(df_train[df_train["Age"] <= 20].mean()).T
df2 = pd.DataFrame(df_train[(df_train["Age"] <= 40) & (20 < df_train["Age"])].mean()).T
df3 = pd.DataFrame(df_train[(40 < df_train["Age"]) & (df_train["Age"] <= 60)].mean()).T
df4 = pd.DataFrame(df_train[60 < df_train["Age"]].mean()).T

df_merge = pd.concat([df1, df2, df3, df4])
df_merge["Age"] = ["0-20", "20-40", "40-60", "60-80"]
df_merge = df_merge[['Survived', 'Age']]


plt.figure()
ax3 = sns.barplot(x="Age", y="Survived", data=df_merge, ci=None)
plt.title('Survival rate by age group')
# plt.show()
plt.close()



data = pd.read_csv("Data/train.csv")
data = data[['Pclass', 'Age', 'Fare', 'Survived', 'Sex']]
data.dropna(inplace=True)
data.reset_index(drop=True, inplace=True)
data["Sex"] = data['Sex'].astype('category').cat.codes



# data = data.drop(['Ticket', 'Name', 'PassengerId', 'Parch', 'Embarked', 'SibSp', 
#                     'Sex', 'Cabin', 'Survived'], axis=1)


y = data["Survived"]  # Variable to explain
data = data.drop(['Survived'], axis=1)  # Explanatory variables


# df_train.select_dtypes(np.number).drop(["PassengerId"], axis=1)



# Logistic Regression using Sklearn


# None penalty and we use a Newton method for the approximation of the coefficient
modele_logit = LogisticRegression(penalty='none', solver='newton-cg')

# We fit the model
modele_logit.fit(X=data, y=y)

# Score 
# print(modele_logit.score(X=data, y=y))

# Probability estimates
# print(modele_logit.predict_proba(X=data))

# Prediction
y_pred_logit = list(modele_logit.predict(X=df_test_2.drop("Sex", axis=1)))


cm_logit = confusion_matrix(y_pred=y_pred_logit, y_true=y_true)
ax_1 = plt.subplot()
sns.heatmap(cm_logit, annot=True, fmt='g', ax=ax_1)
ax_1.set_xlabel('Predicted labels')
ax_1.set_ylabel('True labels') 
ax_1.set_title('Confusion Matrix')
ax_1.xaxis.set_ticklabels(['Not Survived', 'Survived'])
ax_1.yaxis.set_ticklabels(['Not Survived', 'Survived'])
# plt.show()
plt.close()

# print(classification_report(y_pred=y_pred_logit, y_true=y_true))
# F1-score is low for label "Survived" and is high for label "Not Survived"


# We gather all the information in a dataframe
result_1 = pd.DataFrame(np.concatenate([modele_logit.intercept_.reshape(-1, 1),
                             modele_logit.coef_], axis=1),
                             index=["coef"],
                             columns=["constante"]+list(data.columns)).T


# print(result_1)

# print(classification_report(y_pred=y_pred_logit, y_true=y_true))

# Logistic Regression using statsmodels

# We add a constant to the model (the data are not centered)
data_stat = sm.add_constant(data)
y = list(y)
model = sm.Logit(endog=y, exog=data_stat)

# We fit the model
result_2 = model.fit()



# print(result_2.summary())
y_pred_stats = list(result_2.predict(sm.add_constant(df_test_2).drop("Sex", axis=1)))
y_pred_stats = list(map(round, y_pred_stats))



cm_stats = confusion_matrix(y_pred=y_pred_stats, y_true=y_true)
ax_2 = plt.subplot()
sns.heatmap(cm_stats, annot=True, fmt='g', ax=ax_2)
ax_2.set_xlabel('Predicted labels')
ax_2.set_ylabel('True labels') 
ax_2.set_title('Confusion Matrix')
ax_2.xaxis.set_ticklabels(['Not Survived', 'Survived'])
ax_2.yaxis.set_ticklabels(['Not Survived', 'Survived'])
# plt.show()
plt.close()





# Conclusion :

# We see that the results provided by sklearn and statsmodels are almost the same.

# It seems that the class to which the individual belongs, has a significant effect 
# on his chances of survival.
# More precisely, the coefficient (negative) associated to
# the class variable (Pclass), show that the chances of survival are all the more important that the
# individual belongs to a high class.

# The variables Age and Fare did not appear to have a significant role in the chance of survival.


# Some calculation by hand

s = np.array(result_1)
r = data_stat.dot(s)
h = pd.DataFrame(data=list(r[0]), columns=["Proba"])
p = np.exp(h) / (1+np.exp(h))


d = pd.concat([data["Pclass"], p], axis=1)

l = []
c = 0

for i in range(len(y)):

    if p.iloc[i, 0] > 0.5:
        l.append(1)
    else:
        l.append(0)

    if y[i] == l[i]:
        c += 1

score_train_set = c/len(y)

# print(score_train_set)
# Logistic regression correctly predicts 79% for the train set (taking a threshold
# of 0.5 for the probability of attribution)



pred_real = pd.DataFrame(data=[y, l], index=['Real', 'Predict']).T
# print(pred_real)

confusion = pd.crosstab(pred_real["Real"], pred_real["Predict"])
# print(confusion)

yy = np.array(y)
ll = np.array(l)

# Other way
# print(pd.crosstab(index=yy, columns=ll, rownames=["Real"], colnames=["Predict"]))



df_test_3 = pd.read_csv("Data/test.csv")
df_test_3 = df_test_3[["Age", "Pclass", "Fare", "Sex"]]
df_test_3["Sex"] = df_test_3["Sex"].astype('category').cat.codes
df_test_3.dropna(inplace=True)
df_test_3.reset_index(drop=True, inplace=True)
s1 = np.array(result_1)
r1 = sm.add_constant(df_test_3).dot(s1)
p1 = np.exp(r1) / (1+np.exp(r1))


df_test_4 = pd.read_csv("Data/gender_submission.csv")
df_test_4 = df_test_4.loc[df_test_3.index, :]
df_test_4.dropna(inplace=True)
df_test_4.reset_index(drop=True, inplace=True)


y1 = list(df_test_4["Survived"])

l1 = []
c1 = 0


for i in range(len(y1)):

    if p1.iloc[i, 0] > 0.5:
        l1.append(1)
    else:
        l1.append(0)

    if y1[i] == l1[i]:
        c1 += 1

score_test_set = c1/len(y1)

# print(score_test_set)
# Logistic regression correctly predicts 63% for the test set (taking a threshold 
# of 0.5 for the probability of attribution)



# Logistics functions

plt.figure()
plt.scatter(r1.iloc[:,  0], p1.iloc[:, 0])
plt.title(label="Logistic function on test data")
plt.xlabel(xlabel="x (individuals)")
plt.ylabel(ylabel="survival probability")
# plt.show()
plt.close()


plt.figure()
plt.scatter(h.iloc[:, 0], p.iloc[:, 0])
plt.scatter(np.zeros(d[d.loc[:, "Pclass"] == 1]["Proba"].shape[0])-3.2, d[d.loc[:, "Pclass"] == 1]["Proba"], c="purple", label="Class 1", marker=4)
plt.scatter(np.zeros(d[d.loc[:, "Pclass"] == 2]["Proba"].shape[0])-3.4, d[d.loc[:, "Pclass"] == 2]["Proba"], c="red", label="Class 2", marker=4)
plt.scatter(np.zeros(d[d.loc[:, "Pclass"] == 3]["Proba"].shape[0])-3.6, d[d.loc[:, "Pclass"] == 3]["Proba"], c="green", label="Class 3", marker=4)
plt.title(label="Logistic function on train data")
plt.xlabel(xlabel="x (individuals)")
plt.ylabel(ylabel="survival probability")
plt.legend()
# plt.show()
plt.close()


#----------------------------------------------------------------#

# kneighborsclassifier'method (KNN)

model_knn = KNeighborsClassifier(n_neighbors=2)
result_3 = model_knn.fit(X=data, y=y)

# print(model_knn.score(X=data, y=y))
# print(model_knn.score(X=df_test_3, y=y_true))

# 79% of the values were correctly predicted on the train set
# 44% of the values were correctly predicted on the test set

# print(model_knn.predict_proba(X=data))
# print(model_knn.predict_proba(X=df_test_3))


# print(model_knn.predict(X=data))
# print(model_knn.predict(X=df_test_3))

y_pred_knn = model_knn.predict(X=df_test_3)

# Prediction for me...
# df_test_3 = df_test_3.append(pd.DataFrame(np.array([22, 3, 7, 1]).reshape(1, 4), columns=list(df_test_3.columns)),
#                                 ignore_index=True)
# y_pred_knn = model_knn.predict(X=df_test_3)
# print(y_pred_knn[-1])

cm_knn = confusion_matrix(y_pred=y_pred_knn, y_true=y_true)
ax_3 = plt.subplot()
sns.heatmap(cm_knn, annot=True, fmt='g', ax=ax_3)
ax_3.set_xlabel('Predicted labels')
ax_3.set_ylabel('True labels') 
ax_3.set_title('Confusion Matrix')
ax_3.xaxis.set_ticklabels(['Not Survived', 'Survived'])
ax_3.yaxis.set_ticklabels(['Not Survived', 'Survived'])
# plt.show()
plt.close()

# print(classification_report(y_pred=y_pred_knn, y_true=y_true))
# F1-score is low (for both labels)

scores_train = []
scores_test = []

for k in range(1, 10):

   classifier = KNeighborsClassifier(n_neighbors=k)
   classifier.fit(X=data, y=y)
   scores_train.append(classifier.score(X=data, y=y))
   y_pred = classifier.predict(X=df_test_3)
   scores_test.append(accuracy_score(y_true=y_true, y_pred=y_pred))


plt.figure()
plt.plot(range(1, 10), scores_train, c="red", label="Train")
plt.plot(range(1, 10), scores_test, c="purple", label="Test")
plt.xlabel("Value of k")
plt.ylabel("Accuracy")
plt.legend()
# plt.show()
plt.close()

# The best k seems to be k=1 (score: 98%) for the train set
# The best compromise between the train and test scores seems to be k=2
