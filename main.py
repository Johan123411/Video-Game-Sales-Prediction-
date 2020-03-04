######  AUTHOR : SIDDHANT BARUA  ######

import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import warnings
import statistics
import time
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.svm import SVR
from sklearn.metrics import accuracy_score

n = 15  # number of iterations of the training and testing process
scaler = StandardScaler()
linear_regression = LinearRegression()
forest = RandomForestRegressor()
boosting = GradientBoostingRegressor(random_state=60)
regressor = SVR(
    kernel='linear')  # kernels are rbf, linear and polynomial out of which polynomial kernel gives the highest MAE and MSE

warnings.filterwarnings("ignore")
pd.set_option('display.width', 10000000)
pd.set_option('display.max_columns', 10000000)
# pd.set_option('display.max_rows', 10000000)

DataSet = pd.read_csv("Video_Games_Sales_as_at_22_Dec_2016.csv")


# Wii sports is not a game, it's bundle of games that sold arround 82.53 million
# copies, which is much higher than any other game in the dataset,
# This will be a huge outlier and hence affects accuracy of any model we hence remove it
DataSet.drop(index=[0], inplace=True)
DataSet.drop('Developer', axis=1, inplace=True)  # Dropping developer column because it has way to many unique non
# numerioc values
# print(DataSet)
# print(DataSet.shape)
# print(DataSet.info())

# For real-valued input, log1p is accurate also for X so small that 1 + X == 1 in floating-point accuracy.
# Src: https://stackoverflow.com/questions/49538185/what-is-the-purpose-of-numpy-log1p
DataSet['Global_Sales_Log'] = np.log1p(DataSet['Global_Sales'])
DataSet['NA_Sales_Log'] = np.log1p(DataSet['NA_Sales'])
DataSet['EU_Sales_Log'] = np.log1p(DataSet['EU_Sales'])
DataSet['JP_Sales_Log'] = np.log1p(DataSet['JP_Sales'])
DataSet['Other_Sales_Log'] = np.log1p(DataSet['Other_Sales'])

# Evaluating missing data
# SRC: https://chartio.com/resources/tutorials/how-to-check-if-any-value-is-nan-in-a-pandas-dataframe/

# null_columns = DataSet.columns[DataSet.isnull().any()]
# print(DataSet[null_columns].isnull().sum())

# Filling all NaN values with 0
DataSet.Year_of_Release = DataSet.Year_of_Release.fillna(0)


# Printing out all unique gaming platforms
# print(DataSet[DataSet['Year_of_Release'] == 0].Platform.unique())

# Now we find out the median values of the Year_of_Release column for all unique platforms
# SRC: https://pandas.pydata.org/pandas-docs/stable/reference/api/pandas.DataFrame.median.html
# Now the intuition is to change all the values previously set to zero by the median value of Year_of_release for
# that particular platform

def medYear(X):
    if X.Year_of_Release == 0:
        if X.Platform == 'PS2':
            return DataSet[DataSet['Platform'] == 'PS2']['Year_of_Release'].median()
        elif X.Platform == 'Wii':
            return DataSet[DataSet['Platform'] == 'Wii']['Year_of_Release'].median()
        elif X.Platform == '2600':
            return DataSet[DataSet['Platform'] == '2600']['Year_of_Release'].median()
        elif X.Platform == 'X360':
            return DataSet[DataSet['Platform'] == 'X360']['Year_of_Release'].median()
        elif X.Platform == 'GBA':
            return DataSet[DataSet['Platform'] == 'GBA']['Year_of_Release'].median()
        elif X.Platform == 'PC':
            return DataSet[DataSet['Platform'] == 'PC']['Year_of_Release'].median()
        elif X.Platform == 'PS3':
            return DataSet[DataSet['Platform'] == 'PS3']['Year_of_Release'].median()
        elif X.Platform == 'PS':
            return DataSet[DataSet['Platform'] == 'PS']['Year_of_Release'].median()
        elif X.Platform == 'PSP':
            return DataSet[DataSet['Platform'] == 'PSP']['Year_of_Release'].median()
        elif X.Platform == 'XB':
            return DataSet[DataSet['Platform'] == 'XB']['Year_of_Release'].median()
        elif X.Platform == 'GB':
            return DataSet[DataSet['Platform'] == 'GB']['Year_of_Release'].median()
        elif X.Platform == 'DS':
            return DataSet[DataSet['Platform'] == 'DS']['Year_of_Release'].median()
        elif X.Platform == '3DS':
            return DataSet[DataSet['Platform'] == '3DS']['Year_of_Release'].median()
        elif X.Platform == 'N64':
            return DataSet[DataSet['Platform'] == 'N64']['Year_of_Release'].median()
        elif X.Platform == 'PSV':
            return DataSet[DataSet['Platform'] == 'PSV']['Year_of_Release'].median()
        elif X.Platform == 'GC':
            return DataSet[DataSet['Platform'] == 'GC']['Year_of_Release'].median()
        else:
            return 1900
    else:
        return X.Year_of_Release


# applying the medYear method for all entries in the dataset in the column Year_of_Release
DataSet.Year_of_Release = DataSet.apply(medYear, axis=1)
DataSet.drop(index=[659, 14246], inplace=True)
# in positions 659 and 14246, we remove the missing 2 fields in Name and Genre manually
# null_columns = DataSet.columns[DataSet.isnull().any()]
# print(DataSet[null_columns].isnull().sum())

# Now we move on to the Publisher column, where there are 54 missing entries
DataSet['Publisher'].fillna(value='Unknown', inplace=True)

# The most amount of missing values can be found in Critic_score and Critic_count.
DataSet.Critic_Score = DataSet.Critic_Score.fillna(DataSet.Critic_Score.median())
DataSet.Critic_Count = DataSet.Critic_Count.fillna(DataSet.Critic_Count.median())

# We are now going to deal with the User_Score and User_Count missing values.
DataSet.User_Score = DataSet.User_Score.fillna(0)
DataSet.User_Count = DataSet.User_Count.fillna(DataSet.User_Count.median())
# print(DataSet)

# Some fields of User Score and User count will have tbd values, which we need to standardize,
# in this case we are going to replace this score with 100 this is because in the dataset there is,
# a large amount of TBD and NAN values, since we are already making all the NAN values to 0,
# We need to replace TBD with 100s so that the median is about equal

DataSet.User_Score.replace(to_replace='tbd', value=100, inplace=True)

# after this we replace all 0's and 100's with the median value obtained from replacing NaN
# and tbd fields with 0's amd 100's.

DataSet.replace({'User_Score': {0: DataSet.User_Score.median(), 100: DataSet.User_Score.median()}}, inplace=True)

# We now replace the missing values of User_Rating with Void.
DataSet.Rating.fillna('Void', inplace=True)


#
# print(DataSet.info())

# Now we rank publishers on the basis of how many titles they have released
# We replace everything other than the top 10 with "Other Devs", this is done to standardize
# this column, as there are multiple publishers that have published juut 1 game, and learning models,
# would be needed to be overfit, as there will be extremely distinct outliers

def filterPublisher(X):
    if X.Publisher in DataSet['Publisher'].value_counts()[:10].keys().tolist():
        return X.Publisher
    else:
        return 'Other_Dev'


# Finalizing changes in the dataset


DataSet.Publisher = DataSet.apply(filterPublisher, axis=1)
# print(DataSet)

# We now convert all numerical datatypes from decimal to integer

DataSet['Year_of_Release'] = DataSet.Year_of_Release.astype(int)
DataSet['Critic_Score'] = DataSet.Critic_Score.astype(int)
DataSet['Critic_Count'] = DataSet.Critic_Count.astype(int)
DataSet['User_Count'] = DataSet.User_Count.astype(int)
DataSet.User_Score = pd.to_numeric(DataSet.User_Score, errors='coerce')

# print(DataSet.Platform.unique())
Console_list = []
for platform in DataSet['Platform']:
    if platform == 'PS3' or platform == 'PS2' or platform == 'PS' or platform == 'PS4':
        Console_list.append('Playstation')
    elif platform == 'X360' or platform == 'XB' or platform == 'XOne':
        # Append 'XBOX'
        Console_list.append('XBOX')
        # else, if row in Nintendo list
    elif platform == 'NES' or platform == 'Wii' or platform == 'GC' or platform == 'SNES' or platform == 'N64' or platform == 'WiiU':
        # Append 'Nintendo'
        Console_list.append('Nintendo')
        # otherwise,
    else:
        # Append a failing grade
        Console_list.append('Other_Platform')

DataSet["Console_Universal_Set"] = Console_list
# print(DataSet)

# We need to find out the age of the game instead of the year it was released
# Age of the game can be thought of as Game_Release_Date - Release_Date_Of_The_Console.
DataSet["Age"] = ''


def age(X):
    if X.Platform == 'PS2':
        return X.Year_of_Release - 2000
    elif X.Platform == 'Wii':
        return X.Year_of_Release - 2006
    elif X.Platform == '2600':
        return X.Year_of_Release - 1977
    elif X.Platform == 'X360':
        return X.Year_of_Release - 2005
    elif X.Platform == 'GBA':
        return X.Year_of_Release - 2001
    elif X.Platform == 'PS3':
        return X.Year_of_Release - 2006
    elif X.Platform == 'PS':
        return X.Year_of_Release - 1994
    elif X.Platform == 'PSP':
        return X.Year_of_Release - 2004
    elif X.Platform == 'XB':
        return X.Year_of_Release - 2001
    elif X.Platform == 'GB':
        return X.Year_of_Release - 1989
    elif X.Platform == 'DS':
        return X.Year_of_Release - 2004
    elif X.Platform == '3DS':
        return X.Year_of_Release - 2011
    elif X.Platform == 'N64':
        return X.Year_of_Release - 1996
    elif X.Platform == 'PSV':
        return X.Year_of_Release - 2011
    elif X.Platform == 'GC':
        return X.Year_of_Release - 2001
    else:
        return 1


DataSet.Age = DataSet.apply(age, axis=1)
# print(DataSet[DataSet["Age"]<0])

# We notice that one game has value -19 which is another Outlier we drop this
# We also replace all -1's with 0's as most likely these games were released along with the console

DataSet.drop(index=[15959], inplace=True)
DataSet.Age.loc[[1340, 2076, 12301]] = 0

DataSet.drop('Year_of_Release', axis=1, inplace=True)

# print(DataSet)

dummie_data = pd.get_dummies(DataSet[['Platform', 'Genre', 'Publisher', 'Rating']], drop_first=True)
DataSet = pd.merge(dummie_data, DataSet, left_index=True, right_index=True)
DataSet.drop(['Name', 'Platform', 'Genre', 'Publisher', 'Rating', 'Console_Universal_Set'], axis=1, inplace=True)

# print(DataSet.head())

Standardized_features = scaler.fit_transform(
    DataSet[['Critic_Score', 'Critic_Count', 'User_Score', 'User_Count', 'Age']])
Standardized_data = pd.DataFrame(Standardized_features,
                                 columns=['Critic_Score', 'Critic_Count', 'User_Score', 'User_Count', 'Age'])

# Merge scaled_data with the original dataset on index
DataSet = pd.merge(Standardized_data, DataSet, left_index=True, right_index=True)

# print(DataSet)
# Drop initial non-standardized features
DataSet.drop(['Critic_Score_y', 'Critic_Count_y', 'User_Score_y', 'User_Count_y', 'NA_Sales', 'EU_Sales', 'JP_Sales',
              'Other_Sales', 'Global_Sales', 'Age_y'], inplace=True, axis=1)

# We are going to split training & test data now
# X = training data
# Y = target data i.e. what er try and predict
temp = DataSet
MAE_list_LR = []
MSE_list_LR = []
MAE_list_RF = []
MSE_list_RF = []
MAE_list_B = []
MSE_list_B = []
MAE_list_SVM = []
MSE_list_SVM = []
time_list_LR = []
time_list_RF = []
time_list_B = []
time_list_SVM = []



for i in range(n):
    DataSet = temp.reindex(np.random.permutation(temp.index))
    X = DataSet.drop(['Global_Sales_Log', 'NA_Sales_Log', 'EU_Sales_Log', 'JP_Sales_Log', 'Other_Sales_Log'], axis=1)
    y = DataSet['Global_Sales_Log']
    # We notice that the features from column 5 onwards to the end
    X = DataSet.drop(DataSet.columns[5:64], axis=1, inplace=True)
    X = DataSet.drop(['Global_Sales_Log', 'EU_Sales_Log', 'JP_Sales_Log', 'Other_Sales_Log'], axis=1)
    #
    # print(y.head())
    # exit()
    X_trainingSet, X_testingSet, y_trainingSet, y_testingSet = train_test_split(X, y, test_size=0.33, random_state=101)

    ######## LINEAR REGRESSION #########
    t0 = time.clock()
    linear_regression.fit(X_trainingSet, y_trainingSet)
    Regression_predictions = linear_regression.predict(X_testingSet)
    t1 = time.clock()
    time_list_LR.append((t1-t0))


    MAE_LRegression = mean_absolute_error(y_testingSet, Regression_predictions)
    MSE_LRegression = mean_squared_error(y_testingSet, Regression_predictions)


    MAE_list_LR.append(MAE_LRegression)
    MSE_list_LR.append(MSE_LRegression)

    if i == (n - 1):
        print(" ------------------ LINEAR REGRESSION ------------------ ")
        print("Mean Absolute Error (MAE) :" + str(MAE_LRegression))
        print("Mean Squared Error (MSE) :" + str(MSE_LRegression))

        # Graphs

        sns.scatterplot(Regression_predictions, y_testingSet)
        plt.title("Linear Regression", fontdict=None, loc='center', pad=None)
        plt.show()
        plt.scatter(np.squeeze(Regression_predictions), y_testingSet)
        plt.xlabel('Predictions')
        plt.ylabel('Target Values')
        plt.xlim(plt.xlim())
        plt.ylim(plt.ylim())
        plt.plot(np.arange(10))
        _ = plt.plot([-100, 100], [-100, 100], 'Red')
        plt.title("Linear Regression", fontdict=None, loc='center', pad=None)
        plt.show()

    ######## RANDOM FOREST ##########
    t0 = time.clock()
    forest.fit(X_trainingSet, y_trainingSet)
    forest_predictions = forest.predict(X_testingSet)
    t1 = time.clock()
    time_list_RF.append((t1 - t0))

    MAE_random_forest = mean_absolute_error(y_testingSet, forest_predictions)
    MSE_random_forest = mean_squared_error(y_testingSet, forest_predictions)

    MAE_list_RF.append(MAE_random_forest)
    MSE_list_RF.append(MSE_random_forest)

    if i == (n - 1):
        print(" ------------------ RANDOM FOREST ------------------ ")
        print("Mean Absolute Error (MAE) :" + str(MAE_random_forest))
        print("Mean Squared Error (MSE) :" + str(MSE_random_forest))

        # Graphs
        sns.scatterplot(forest_predictions, y_testingSet)
        plt.title("Random Forest", fontdict=None, loc='center', pad=None)
        plt.show()
        plt.scatter(np.squeeze(forest_predictions), y_testingSet)
        plt.xlabel('Predictions')
        plt.ylabel('Target Values')
        plt.xlim(plt.xlim())
        plt.ylim(plt.ylim())
        plt.plot(np.arange(10))
        _ = plt.plot([-100, 100], [-100, 100], 'Red')
        plt.title("Random Forest", fontdict=None, loc='center', pad=None)
        plt.show()

    ######## BOOSTING ##########

    # here we use gradient boosting
    t0 = time.clock()
    boosting.fit(X_trainingSet, y_trainingSet)
    boosting_pred = boosting.predict(X_testingSet)
    t1 = time.clock()
    time_list_B.append((t1 - t0))

    MAE_boosting = mean_absolute_error(y_testingSet, boosting_pred)
    MSE_boosting = mean_squared_error(y_testingSet, boosting_pred)


    MAE_list_B.append(MAE_boosting)
    MSE_list_B.append(MSE_boosting)

    if i == (n - 1):
        print(" ------------------ BOOSTING ------------------ ")
        print("Mean Absolute Error (MAE) :" + str(MAE_boosting))
        print("Mean Squared Error (MSE) :" + str(MSE_boosting))

        # Graphs
        sns.scatterplot(boosting_pred, y_testingSet)
        plt.title("Boosting", fontdict=None, loc='center', pad=None)
        plt.show()
        plt.scatter(np.squeeze(boosting_pred), y_testingSet)
        plt.xlabel('Predictions')
        plt.ylabel('Target Values')
        plt.xlim(plt.xlim())
        plt.ylim(plt.ylim())
        plt.plot(np.arange(10))
        _ = plt.plot([-100, 100], [-100, 100], 'Red')
        plt.title("Boosting", fontdict=None, loc='center', pad=None)
        plt.show()

    ######## SVM ##########
    t1 = time.clock()
    regressor.fit(X_trainingSet, y_trainingSet)
    SVM_pred = regressor.predict(X_testingSet)
    t1 = time.clock()
    time_list_SVM.append((t1 - t0))

    MAE_SVM = mean_absolute_error(y_testingSet, SVM_pred)
    MSE_SVM = mean_squared_error(y_testingSet, SVM_pred)

    MAE_list_SVM.append(MAE_SVM)
    MSE_list_SVM.append(MSE_SVM)

    if i == (n - 1):
        print(" ------------------ SVM ------------------ ")
        print("Mean Absolute Error (MAE) :" + str(MAE_SVM))
        print("Mean Squared Error (MSE) :" + str(MSE_SVM))

        # Graphs
        sns.scatterplot(SVM_pred, y_testingSet)
        plt.title("SVM", fontdict=None, loc='center', pad=None)
        plt.show()
        plt.scatter(np.squeeze(SVM_pred), y_testingSet)
        plt.xlabel('Predictions')
        plt.ylabel('Target Values')
        plt.xlim(plt.xlim())
        plt.ylim(plt.ylim())
        plt.plot(np.arange(10))
        _ = plt.plot([-100, 100], [-100, 100], 'Red')
        plt.title("SVM", fontdict=None, loc='center', pad=None)
        plt.show()

MAE_Avg_LR = statistics.mean(MAE_list_LR)
MSE_Avg_LR = statistics.mean(MSE_list_LR)

MAE_Avg_RF = statistics.mean(MAE_list_RF)
MSE_Avg_RF = statistics.mean(MSE_list_RF)

MAE_Avg_B = statistics.mean(MAE_list_B)
MSE_Avg_B = statistics.mean(MSE_list_B)

MAE_Avg_SVM = statistics.mean(MAE_list_SVM)
MSE_Avg_SVM = statistics.mean(MSE_list_SVM)

############### MODEL COMPARISONS ######################

# MAE's
def showMAE():
    plt.figure(figsize=(20, 8))
    sns.barplot(['Linear Regression', 'Random Forest', 'Boosting', 'SVM'],
                [MAE_Avg_LR, MAE_Avg_RF, MAE_Avg_B, MAE_Avg_SVM], palette='husl')
    plt.title("MAE Comparison Graph", fontdict={'fontsize': 30}, loc='center', pad=None, )
    plt.show()
# MSE's
def showMSE():
    plt.figure(figsize=(20, 8))
    sns.barplot(['Linear Regression', 'Random Forest', 'Boosting', 'SVM'],
                [MSE_Avg_LR, MSE_Avg_RF, MSE_Avg_B, MSE_Avg_SVM], palette='husl')
    plt.title("MSE Comparison Graph", fontdict={'fontsize': 30}, loc='center', pad=None)
    plt.show()
# Time
def showTime():
    plt.figure(figsize=(20, 8))
    sns.barplot(['Linear Regression', 'Random Forest', 'Boosting', 'SVM'],
                [statistics.mean(time_list_LR), statistics.mean(time_list_RF), statistics.mean(time_list_B), statistics.mean(time_list_SVM)], palette='husl')
    plt.title("Time Comparison Graph", fontdict={'fontsize': 30}, loc='center', pad=None)
    plt.show()

showMAE()
showMSE()
showTime()

print("==============================================")
print("the average time taken for Linear Regression ", statistics.mean(time_list_LR)) # since time taken by Linear Regression is so small we print it out
# print(time_list_RF)
# print(time_list_B)
# print(time_list_SVM)
