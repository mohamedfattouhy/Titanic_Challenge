import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from random import random

def uniforme(a,b):
    return(a+(b-a)*random())

data_train_raw = pd.read_csv("Data/test.csv")
truth = pd.read_csv("Data/gender_submission.csv")
print(data_train_raw.head())

def prediciton(n):
    prediction_accuracy = []
    for j in range(n):
        data_survived =  []

        for i in range(418):
            if data_train_raw.iloc[i, 4] is (float or int) :
                
                if data_train_raw.iloc[i, 3] == "male":

                    if data_train_raw.iloc[i, 1] == 1:
                        if data_train_raw.iloc[i, 4] < 15:
                            data_survived.append(1)
                        else:
                            if uniforme(0, 1) < 0.37:
                                data_survived.append(1)
                            else:
                                data_survived.append(0)


                    if data_train_raw.iloc[i, 1] == 2:
                        if data_train_raw.iloc[i, 4] < 15:
                            data_survived.append(1)
                        else:
                            if uniforme(0, 1) < 0.068:
                                data_survived.append(1)
                            else:
                                data_survived.append(0)
                    if data_train_raw.iloc[i, 1] == 3:
                        if data_train_raw.iloc[i, 4] < 15:
                            if uniforme(0, 1) < 0.3333:
                                data_survived.append(1)
                            else:
                                data_survived.append(0)
                        else:
                            if uniforme(0, 1) < 0.128:
                                data_survived.append(1)
                            else:
                                data_survived.append(0)
                else:
                    if data_train_raw.iloc[i, 1] == 1:
                        if data_train_raw.iloc[i, 4] < 15:
                            if uniforme(0, 1) < 0.5:
                                data_survived.append(1)
                            else:
                                data_survived.append(0)
                        else:
                            if uniforme(0, 1) < 0.975:
                                data_survived.append(1)
                            else:
                                data_survived.append(0)

                    if data_train_raw.iloc[i, 1] == 2:
                        if uniforme(0, 1) < 0.921:
                            data_survived.append(1)
                        else:
                            data_survived.append(0)
                    
                    if data_train_raw.iloc[i, 1] == 3:
                        if data_train_raw.iloc[i, 4] < 15:
                            if uniforme(0, 1) < 0.5:
                                data_survived.append(1)
                            else:
                                data_survived.append(0)
                        else:
                            if uniforme(0, 1) < 0.45:
                                data_survived.append(1)
                            else:
                                data_survived.append(0)

            else:
                
                if data_train_raw.iloc[i, 3] == "male":
                    if data_train_raw.iloc[i, 1] == 1:
                        if uniforme(0, 1) < 0.368:
                            data_survived.append(1)
                        else:
                            data_survived.append(0)

                    if data_train_raw.iloc[i, 1] == 2:
                        if uniforme(0, 1) < 0.157:
                            data_survived.append(1)
                        else:
                            data_survived.append(0)
                    
                    if data_train_raw.iloc[i, 1] == 3:
                        if uniforme(0, 1) < 0.1354:
                            data_survived.append(1)
                        else:
                            data_survived.append(0)
                else:
                    if data_train_raw.iloc[i, 1] == 1:
                        if uniforme(0, 1) < 0.968:
                            data_survived.append(1)
                        else:
                            data_survived.append(0)

                    if data_train_raw.iloc[i, 1] == 2:
                        if uniforme(0, 1) < 0.921:
                            data_survived.append(1)
                        else:
                            data_survived.append(0)
                    
                    if data_train_raw.iloc[i, 1] == 3:
                        if uniforme(0, 1) < 0.5:
                            data_survived.append(1)
                        else:
                            data_survived.append(0)
        counter = 0
        for i in range(418):
            if data_survived[i] == truth.iloc[i, 1]:
                counter = counter + 1
        
        prediction_accuracy.append(counter/418)


    return prediction_accuracy



print(sum(prediciton(100))/100)
print('lol')

# 78% sur les données de test