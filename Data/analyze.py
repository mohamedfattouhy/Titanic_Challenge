import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns



df = pd.read_csv("Data/train.csv")
df = df.drop(['Ticket', 'Fare', 'Cabin'], axis=1)
df.dropna(inplace=True)
df.index = [i for i in range(df.shape[0])]
# 712 rows and 9 columns

# Short description
# print(df.describe())


df.groupby('Sex')[['Survived']].aggregate(lambda x: x.mean())
# Women are more chance to survive than men in general


# ax = sns.barplot(x="Pclass", y="Survived", hue="Sex", data=df)
# plt.title('Survival rate per sex and class')
# plt.show()

# It is clear that women are more likely to survive than men, regardless of class.
# Likewise, the chances of survival are higher when you come from the first class.

# 259 women and 195 survived
# 453 men and 93 survived


ax1 = sns.barplot(x="Embarked", y="Survived", hue="Sex", data=df)
plt.title('Survival rate per Survival rates by boat port')

# Those who embarked at port C, have a better chance of survival
# The rate is roughly equivalent for the other two ports (S and Q)


ax2 = sns.barplot(x="Age", y="Survived", data=df)
plt.title('Survival rate per age')

# Those between the ages of 30 and 50 are less likely to die



def prediction(Pclass, Age, Sex):

    survived = 0

    if Pclass == 1:
        if Sex == "male" and Age <= 50 and Age >= 30 : survived = 1
        if Sex == "female" and Age <= 70 and Age >= 20: survived = 1
        else: survived = 0

    if Pclass == 2 or  Pclass == 3:
        if Sex == "male": survived = 0
        if Sex == "female": survived = 1
        else: survived = 0

    return survived


def prediction_accuracy(data_1, data_2):

    count = 0

    for i in range(data_1.shape[0]):
        if data_1.loc[i, "Survived"] == prediction(Pclass=data_2.loc[i,"Pclass"], Age=data_2.loc[i,"Age"], Sex=data_2.loc[i,"Sex"]): count += 1


    return f'Accuracy = {count/data_1.shape[0]}'



df_test_1 = pd.read_csv("Data/gender_submission.csv")
df_test_2 = pd.read_csv("Data/test.csv")

print(prediction_accuracy(data_1=df_test_1, data_2=df_test_2))

# 98% of the values were correctly predicted