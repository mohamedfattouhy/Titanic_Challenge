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
test1 = test1.dropna()
print(test1.shape)
print(test1)


first_male = 0
survivied_first_male = 0

survivied_first_male_young = 0
died_first_male_young = 0
survivied_first_male_mid = 0
died_first_male_mid = 0
survivied_first_male_old = 0
died_first_male_old = 0





first_female = 0
survivied_first_female = 0


second_male = 0 
survivied_second_male = 0
survivied_second_male_young = 0
died_second_male_young = 0
survivied_second_male_mid = 0
survivied_second_male_old = 0

second_female = 0 
survivied_second_female = 0

third_male = 0 
survivied_third_male = 0
third_female = 0
survivied_third_female = 0

survived_female = 0
survived_male = 0

for i in range(714):

    
    if test1.iloc[i, 0] == 1:

        # Counter for first class

        if test1.iloc[i, 2] == 'male':

            # Counter for male in first class

            first_male = first_male + 1

            if test1.iloc[i, 3] == 1:

                # Counter for male in first class who survived

                survivied_first_male += 1

                if test1.iloc[i, 1] < 15:

                    survivied_first_male_young += 1

                if test1.iloc[i, 1] > 35:

                    survivied_first_male_old += 1

                if 15 < test1.iloc[i, 1] < 35:

                    survivied_first_male_mid += 1
            
            else:

                if test1.iloc[i, 1] < 15:

                    died_first_male_young += 1

                if test1.iloc[i, 1] > 35:

                    died_first_male_old += 1

                if 15 < test1.iloc[i, 1] < 35:

                    died_first_male_mid += 1
                
        else:
            first_female = first_female + 1
            if test1.iloc[i, 3] == 1:
                survivied_first_female = survivied_first_female + 1
   
    if test1.iloc[i, 0] == 2:
        if test1.iloc[i, 2] == 'male':
            second_male = second_male + 1
            if test1.iloc[i, 3] == 1:
                survivied_second_male = survivied_second_male + 1
        else:
            second_female = second_female + 1
            if test1.iloc[i, 3] == 1:
                survivied_second_female = survivied_second_female + 1

    if test1.iloc[i, 0] == 3:
        if test1.iloc[i, 2] == 'male':
            third_male = third_male + 1
            if test1.iloc[i, 3] == 1:
                survivied_third_male = survivied_third_male + 1
        else:
            third_female = third_female + 1
            if test1.iloc[i, 3] == 1:
                survivied_third_female = survivied_third_female + 1


    if test1.iloc[i, 3] == 1:
        if test1.iloc[i, 2] == 'male':
            survived_male = survived_male + 1
        else:
            survived_female = survived_female + 1

    


# print(f'Number of male first class {first_male}')
# print(f'Number of female first class {first_female}')
# print(f'Number of first class {first_male + first_female}')
# print(f'proportion of woman in first class {first_female/(first_male + first_female)}')
# print('')
# print(f'Number of male second class {second_male}')
# print(f'Number of female second class {second_female}')
# print(f'Number of second class {second_male + second_female}')
# print(f'proportion of woman in second class {second_female/(second_male + second_female)}')
# print('')
# print(f'Number of male third class {third_male}')
# print(f'Number of female third class {third_female}')
# print(f'Number of third class {third_male + third_female}')
# print(f'proportion of woman in third class {third_female/(third_male + third_female)}')
# print('')
# print('')
# print(f'proportion of woman {(first_female + second_female + third_female)/(third_male + third_female + second_male + second_female + first_male + first_female)}')
# print('')
# print('')
# print(f'proportion of woman saved: {survived_female/(third_female + first_female + second_female)}')
# print(f'proportion of male saved: {survived_male/(third_male + first_male + second_male)}')


# # plt.figure()
# # ax = sns.violinplot(x="Pclass", y="Age", hue="Sex", data=test1, split=True)
# # plt.title('Age distribution by gender and class (Titanic passengers)')
# # plt.show()

# print(f'survived woman in first class {survivied_first_female/(first_female)}%')
# print(f'survived woman in second class {survivied_second_female/(second_female)}%')
# print(f'survived woman in third class {survivied_third_female/(third_female)}%')


# print(f'survived man in first class {survivied_first_male/(first_male)}%')
# print(f'survived man in second class {survivied_second_male/(second_male)}%')
# print(f'survived man in third class {survivied_third_male/(third_male)}%')


# print(f'proportion de jeune survivant premiere classe {survivied_first_male_young/(survivied_first_male_young + died_first_male_young)} %')
# print(f'proportion de moyen survivant premiere classe {survivied_first_male_mid/(survivied_first_male_mid + died_first_male_mid)} %')
# print(f'proportion de vieux survivant premiere classe {survivied_first_male_old/(survivied_first_male_old + died_first_male_old)} %')


def counter_loop(Pclass, sex, min_age, max_age):
    class_sex = 0 
    died_class_sex_age = 0
    survivied_class_sex_age = 0 
    for i in range(714):
        if test1.iloc[i, 0] == Pclass:

            # Counter for first class

            if test1.iloc[i, 2] == sex:

                # Counter for male in first class

                class_sex = class_sex + 1

                if test1.iloc[i, 3] == 1:

                    # Counter for male in first class who survived

                    if min_age <= test1.iloc[i, 1] < max_age:

                        survivied_class_sex_age += 1
                
                else:

                    if min_age <= test1.iloc[i, 1] < max_age:

                        died_class_sex_age += 1

    return survivied_class_sex_age, died_class_sex_age




A, B = counter_loop(3, 'female', 15, 100)

print(A/(A+B))

