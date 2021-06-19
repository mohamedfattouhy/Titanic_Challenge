import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

data_train_raw = pd.read_csv("Data/train.csv")
print(data_train_raw.head())


# plt.figure(figsize=(5, 5))
# plt.hist(data_train_raw['Age'], density=True, bins=25)
# plt.xlabel('Age')
# plt.ylabel('Proportion')
# plt.title("Passager age histogram")
# plt.show()

# data_age = data_train_raw.dropna(subset=['Age'])

# print(data_age['Age'].mean())
# print(data_age['Age'].median())

print(data_train_raw['Age'].mean())
print(data_train_raw['Age'].median())
print(data_train_raw['Age'].describe())

print(data_train_raw.groupby(['Pclass']).mean())

test1 = pd.concat([data_train_raw['Pclass'], 
                   data_train_raw['Age'],
                   data_train_raw['Sex'], 
                   data_train_raw['Survived']],
                   axis=1)

print(test1.shape)

first_male = 0
first_female = 0
second_male = 0 
second_female = 0 

third_male = 0 
third_female = 0

survived_female = 0
survived_male = 0

for i in range(891):
    if test1.iloc[i, 0] == 1:
        if test1.iloc[i, 2] == 'male':
            first_male = first_male + 1
        else:
            first_female = first_female + 1
    if test1.iloc[i, 0] == 2:
        if test1.iloc[i, 2] == 'male':
            second_male = second_male + 1
        else:
            second_female = second_female + 1

    if test1.iloc[i, 0] == 3:
        if test1.iloc[i, 2] == 'male':
            third_male = third_male + 1
        else:
            third_female = third_female + 1
    if test1.iloc[i, 3] == 1:
        if test1.iloc[i, 2] == 'male':
            survived_male = survived_male + 1
        else:
            survived_female = survived_female + 1

    


print(f'Number of male first class {first_male}')
print(f'Number of female first class {first_female}')
print(f'Number of first class {first_male + first_female}')
print(f'proportion of woman in first class {first_female/(first_male + first_female)}')
print('')
print(f'Number of male second class {second_male}')
print(f'Number of female second class {second_female}')
print(f'Number of second class {second_male + second_female}')
print(f'proportion of woman in second class {second_female/(second_male + second_female)}')
print('')
print(f'Number of male third class {third_male}')
print(f'Number of female third class {third_female}')
print(f'Number of third class {third_male + third_female}')
print(f'proportion of woman in third class {third_female/(third_male + third_female)}')
print('')
print('')
print(f'proportion of woman {(first_female + second_female + third_female)/(third_male + third_female + second_male + second_female + first_male + first_female)}')
print('')
print('')
print(f'proportion of woman saved: {survived_female/(third_female + first_female + second_female)}')
print(f'proportion of male saved: {survived_male/(third_male + first_male + second_male)}')


plt.figure()
ax = sns.violinplot(x="Pclass", y="Age", hue="Sex", data=test1, split=True)
plt.title('Age distribution by gender and class (Titanic passengers)')
plt.show()
