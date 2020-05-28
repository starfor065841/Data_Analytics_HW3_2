import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import OneHotEncoder
from sklearn.model_selection import train_test_split
from sklearn.model_selection import KFold
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.model_selection import RandomizedSearchCV

from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import chi2

data = pd.read_csv('source/online_shoppers_intention.csv')
# print(data.info())
# print(data.shape)
# print(data.describe())
# print(data.isnull().sum())
# print(data.Revenue)
# print(data.head())


labelencoder = LabelEncoder()
data['Month'] = labelencoder.fit_transform(data['Month'])
data['VisitorType'] = labelencoder.fit_transform(data['VisitorType'])
data['Weekend'] = labelencoder.fit_transform(data['Weekend'])
data['Revenue'] = labelencoder.fit_transform(data['Revenue'])

# sns.countplot(data['Revenue'])



# print(data[['Administrative', 'Revenue']].groupby(['Administrative'], as_index=False).mean().sort_values(by='Revenue', ascending=False))
# print(data[['Informational', 'Revenue']].groupby(['Informational'], as_index=False).mean().sort_values(by='Revenue', ascending=False))
# print(data[['SpecialDay', 'Revenue']].groupby(['SpecialDay'], as_index=False).mean().sort_values(by='Revenue', ascending=False))
# print(data[['OperatingSystems', 'Revenue']].groupby(['OperatingSystems'], as_index=False).mean().sort_values(by='Revenue', ascending=False))
# print(data[['Browser', 'Revenue']].groupby(['Browser'], as_index=False).mean().sort_values(by='Revenue', ascending=False))
# print(data[['Region', 'Revenue']].groupby(['Region'], as_index=False).mean().sort_values(by='Revenue', ascending=False))
# print(data[['TrafficType', 'Revenue']].groupby(['TrafficType'], as_index=False).mean().sort_values(by='Revenue', ascending=False))
# print(data[['VisitorType', 'Revenue']].groupby(['VisitorType'], as_index=False).mean().sort_values(by='Revenue', ascending=False))
# print(data[['Weekend', 'Revenue']].groupby(['Weekend'], as_index=False).mean().sort_values(by='Revenue', ascending=False))
# print(data[['Month', 'Revenue']].groupby(['Month'], as_index=False).mean().sort_values(by='Revenue', ascending=False))


# print(data['ProductRelated_Duration'].value_counts())
# B、E高度相關
# print(data['BounceRates'].value_counts())
# print(data['ExitRates'].value_counts())
# print(data['PageValues'].value_counts())
# print(data['SpecialDay'].value_counts())
# print(data['OperatingSystems'].value_counts())
# print(data['Browser'].value_counts())
# print(data['Region'].value_counts())
# print(data['TrafficType'].value_counts())
# print(data['VisitorType'].value_counts())
# print(data['Weekend'].value_counts())

pos = 0
attribute = data.columns.values[0:16]
fig, ax = plt.subplots(4,4,figsize=(20,24))

for attr in attribute:
    pos += 1

    plt.subplot(4,4,pos)
    sns.distplot(data.loc[data['Revenue'] == True][attr], hist=False,label='True')
    sns.distplot(data.loc[data['Revenue'] == False][attr], hist=False,label='False')
    plt.xlabel(attr, fontsize=9)

# 調整子圖間距
plt.subplots_adjust(wspace = 0.5, hspace = 0.5)



# # weekend vs Revenue
# # plt.figure()
# df = pd.crosstab(data['Weekend'], data['Revenue'])
# df.plot(kind = 'bar', stacked = True, figsize = (15, 5), color = ['orange', 'crimson'])
# plt.title('Weekend vs Revenue', fontsize = 30)

# # SpecialDay vs Revenue
# # plt.figure()
# df = pd.crosstab(data['SpecialDay'], data['Revenue'])
# df.plot(kind = 'bar', stacked = True, figsize = (15, 5), color = ['orange', 'crimson'])
# plt.title('SpecialDay vs Revenue', fontsize = 30)

# # OperatingSystems vs Revenue
# # plt.figure()
# df = pd.crosstab(data['OperatingSystems'], data['Revenue'])
# df.plot(kind = 'bar', stacked = True, figsize = (15, 5), color = ['orange', 'crimson'])
# plt.title('OperatingSystems vs Revenue', fontsize = 30)

# # Browser vs Revenue
# # plt.figure()
# df = pd.crosstab(data['Browser'], data['Revenue'])
# df.plot(kind = 'bar', stacked = True, figsize = (15, 5), color = ['orange', 'crimson'])
# plt.title('Browser vs Revenue', fontsize = 30)

# # Region vs Revenue
# # plt.figure()
# df = pd.crosstab(data['Region'], data['Revenue'])
# df.plot(kind = 'bar', stacked = True, figsize = (15, 5), color = ['orange', 'crimson'])
# plt.title('Region vs Revenue', fontsize = 30)

# # TrafficType vs Revenue
# # plt.figure()
# df = pd.crosstab(data['TrafficType'], data['Revenue'])
# df.plot(kind = 'bar', stacked = True, figsize = (15, 5), color = ['orange', 'crimson'])
# plt.title('TrafficType vs Revenue', fontsize = 30)



X = data[(data['Administrative']<=3) | (data['Administrative']==1)].copy()
# # print(X)

# grid = sns.FacetGrid(X, col='Revenue', row='Administrative', height=2.2, aspect=1.6)
# grid.map(plt.hist, 'Administrative_Duration', alpha=.5, bins=20)
# grid.add_legend()


grid = sns.FacetGrid(X, row='Administrative', col='Revenue')
grid.map(plt.hist, 'Administrative_Duration', alpha=.5, bins=20)
grid.add_legend()

# sorted_corrs = data.corr()['Revenue'].sort_values()

# print(sorted_corrs)
# fig, ax = plt.subplots(figsize=(15,10))
# sns.heatmap(data[sorted_corrs.index].corr(), cmap='bwr', annot=True)



# plt.show()

# 4分位數
# quantile_list = [0, .25, .5, .75, 1.]
# quantiles = data['BounceRates'].quantile(quantile_list)
# print(quantiles)

# fig, ax = plt.subplots()
# data['BounceRates'].hist(bins=30, color='#A9C5D3', 
#                              edgecolor='black', grid=False)
# for quantile in quantiles:
#     qvl = plt.axvline(quantile, color='r')
# ax.legend([qvl], ['Quantiles'], fontsize=10)
# ax.set_title('Developer Income Histogram with Quantiles', 
#              fontsize=12)
# ax.set_xlabel('Developer Income', fontsize=12)
# ax.set_ylabel('Frequency', fontsize=12)

# data['BounceRates_log'] = np.log((1+ data['BounceRates']))

# # income_log_mean = np.round(np.mean(data['BounceRates_log']), 2)
# # fig, ax = plt.subplots()
# # data['BounceRates_log'].hist(bins=30, color='#A9C5D3', 
# #                                  edgecolor='black', grid=False)
# # plt.axvline(income_log_mean, color='r')
# # ax.set_title('Developer Income Histogram after Log Transform', 
# #              fontsize=12)
# # ax.set_xlabel('Developer Income (log scale)', fontsize=12)
# # ax.set_ylabel('Frequency', fontsize=12)
# # ax.text(11.5, 450, r'$\mu$='+str(income_log_mean), fontsize=10)

# data['ExitRates_log'] = np.log((1+ data['ExitRates']))

# data['Administrative_DurationBand_log'] = np.log((1+ data['Administrative_Duration']))
# data['Administrative_DurationBand_log'] = np.around(data['Administrative_DurationBand_log'])
# print(data['Administrative_DurationBand_log'])

# data['Informational_DurationBand_log'] = np.log((1+ data['Informational_Duration']))
# data['Informational_DurationBand_log'] = np.around(data['Informational_DurationBand_log'])

# data['ProductRelated_DurationBand_log'] = np.log((1+ data['ProductRelated_Duration']))
# data['ProductRelated_DurationBand_log'] = np.around(data['ProductRelated_DurationBand_log'])



# plt.show()


# log_mean = np.round(np.mean(data['BounceRates_log']), 2)
# data.loc[ data['BounceRates_log'] <= log_mean, 'BounceRates_log'] = 0
# data.loc[ data['BounceRates_log'] > log_mean, 'BounceRates_log'] = 1

# log_mean = np.round(np.mean(data['ExitRates_log']), 2)
# data.loc[ data['ExitRates_log'] <= log_mean, 'ExitRates_log'] = 0
# data.loc[ data['ExitRates_log'] > log_mean, 'ExitRates_log'] = 1

# log_mean = np.round(np.mean(data['Administrative_DurationBand_log']), 2)
# data.loc[ data['Administrative_DurationBand_log'] <= log_mean, 'Administrative_DurationBand_log'] = 0
# data.loc[ data['Administrative_DurationBand_log'] > log_mean, 'Administrative_DurationBand_log'] = 1

# log_mean = np.round(np.mean(data['Informational_DurationBand_log']), 2)
# data.loc[ data['Informational_DurationBand_log'] <= log_mean, 'Informational_DurationBand_log'] = 0
# data.loc[ data['Informational_DurationBand_log'] > log_mean, 'Informational_DurationBand_log'] = 1

# log_mean = np.round(np.mean(data['ProductRelated_DurationBand_log']), 2)
# data.loc[ data['ProductRelated_DurationBand_log'] <= log_mean, 'ProductRelated_DurationBand_log'] = 0
# data.loc[ data['ProductRelated_DurationBand_log'] > log_mean, 'ProductRelated_DurationBand_log'] = 1


# data['BounceRatesBand'] = pd.qcut(data['BounceRates_log'], 4, duplicates='drop')
# print(data[['BounceRatesBand', 'Revenue']].groupby(['BounceRatesBand'], as_index=False).mean().sort_values(by='BounceRatesBand', ascending=True))

# data.loc[ data['BounceRates_log'] <= 0.00311, 'BounceRates_log'] = 0
# data.loc[(data['BounceRates_log'] > 0.00311) & (data['BounceRates_log'] <= 0.0167), 'BounceRates_log'] = 1
# data.loc[ data['BounceRates_log'] > 0.0167, 'BounceRates_log'] = 2
# data['BounceRates_log'] = data['BounceRates_log'].astype(int)

# data = data.drop(['BounceRatesBand'], axis=1)

# data['ExitRatesBand'] = pd.qcut(data['ExitRates_log'], 4, duplicates='drop')
# print(data[['ExitRatesBand', 'Revenue']].groupby(['ExitRatesBand'], as_index=False).mean().sort_values(by='ExitRatesBand', ascending=True))

# data.loc[ data['ExitRates_log'] <= 0.0142, 'ExitRates_log'] = 0
# data.loc[(data['ExitRates_log'] > 0.0142) & (data['ExitRates_log'] <= 0.0248), 'ExitRates_log'] = 1
# data.loc[(data['ExitRates_log'] > 0.0248) & (data['ExitRates_log'] <= 0.0488), 'ExitRates_log'] = 2
# data.loc[ data['ExitRates_log'] > 0.0488, 'ExitRates_log'] = 3
# data['ExitRates_log'] = data['ExitRates_log'].astype(int)

# data = data.drop(['ExitRatesBand'], axis=1)


# data['Administrative_DurationBand'] = pd.qcut(data['Administrative_DurationBand_log'], 4, duplicates='drop')
# print(data[['Administrative_DurationBand', 'Revenue']].groupby(['Administrative_DurationBand'], as_index=False).mean().sort_values(by='Administrative_DurationBand', ascending=True))

# data.loc[ data['Administrative_DurationBand_log'] <= 2.14, 'Administrative_DurationBand_log'] = 0
# data.loc[(data['Administrative_DurationBand_log'] > 2.14) & (data['Administrative_DurationBand_log'] <= 4.546), 'Administrative_DurationBand_log'] = 1
# data.loc[ data['Administrative_DurationBand_log'] > 4.546, 'Administrative_DurationBand_log'] = 2
# data['Administrative_DurationBand_log'] = data['Administrative_DurationBand_log'].astype(int)

# data = data.drop(['Administrative_DurationBand'], axis=1)


# data['Informational_DurationBand'] = pd.qcut(data['Informational_DurationBand_log'], 4, duplicates='drop')
# print(data[['Informational_DurationBand', 'Revenue']].groupby(['Informational_DurationBand'], as_index=False).mean().sort_values(by='Informational_DurationBand', ascending=True))

# data.loc[ data['Informational_DurationBand_log'] <= 7.844, 'Informational_DurationBand_log'] = 0
# data.loc[ data['Informational_DurationBand_log'] > 7.844, 'Informational_DurationBand_log'] = 1
# data['Informational_DurationBand_log'] = data['Informational_DurationBand_log'].astype(int)

# data = data.drop(['Informational_DurationBand'], axis=1)


# data['ProductRelated_DurationBand'] = pd.qcut(data['ProductRelated_DurationBand_log'], 4, duplicates='drop')
# print(data[['ProductRelated_DurationBand', 'Revenue']].groupby(['ProductRelated_DurationBand'], as_index=False).mean().sort_values(by='ProductRelated_DurationBand', ascending=True))

# data.loc[ data['ProductRelated_DurationBand_log'] <= 5.221, 'ProductRelated_DurationBand_log'] = 0
# data.loc[(data['ProductRelated_DurationBand_log'] > 5.221) & (data['ProductRelated_DurationBand_log'] <= 6.397), 'ProductRelated_DurationBand_log'] = 1
# data.loc[(data['ProductRelated_DurationBand_log'] > 6.397) & (data['ProductRelated_DurationBand_log'] <= 7.29), 'ProductRelated_DurationBand_log'] = 2
# data.loc[ data['ProductRelated_DurationBand_log'] > 7.29, 'ProductRelated_DurationBand_log'] = 3
# data['ProductRelated_DurationBand_log'] = data['ProductRelated_DurationBand_log'].astype(int)

# data = data.drop(['ProductRelated_DurationBand'], axis=1)

data = pd.get_dummies(data,columns=['Month', 'VisitorType', 'SpecialDay', 'Region', 'OperatingSystems', 'Weekend'])
print(data.info())


test = data['Revenue']
train = data.drop(['Revenue', 'BounceRates', 'Administrative_Duration', 'Informational_Duration', 'ExitRates', 'ProductRelated_Duration'], axis=1)

# bestchoise = SelectKBest(score_func=chi2, k=8)
# fit = bestchoise.fit(train, test)
# score = pd.DataFrame(fit.scores_)
# column = pd.DataFrame(train.columns)
# featurescore = pd.concat([column, score], axis=1)
# featurescore.columns = ['feature', 'score']
# print(featurescore.nlargest(8, 'score'))

# sorted_corrs = data.corr()['Revenue'].sort_values()

# print(sorted_corrs)
# fig, ax = plt.subplots(figsize=(15,10))
# sns.heatmap(data[sorted_corrs.index].corr(), cmap='RdBu', annot=True)
# plt.show()


# # train = train.drop(['ExitRates'], axis=1)

# 切取訓練集、驗證集
# X_train, last_train, y_train, last_test = train_test_split(train,
#                                                     test, test_size = 0.2, 
#                                                     random_state = 2)


# 切取訓練集、驗證集
X_train, X_test, y_train, y_test = train_test_split(train,
                                                    test, test_size = 0.2, 
                                                    random_state = 2)

kf = KFold(n_splits=5,                                # 設定 K 值
           random_state=1012,
           shuffle=True)
kf.get_n_splits(X_train)                              # 給予資料範圍

train_acc_list = []                                   # 儲存每次訓練模型的準確度
valid_acc_list = []                                   # 儲存每次驗證模型的準確度

for train_index, valid_index in kf.split(X_train):    # 每個迴圈都會產生不同部份的資料
    train_x_split = X_train.iloc[train_index]         # 產生訓練資料
    train_y_split = y_train.iloc[train_index]         # 產生訓練資料標籤
    valid_x_split = X_train.iloc[valid_index]         # 產生驗證資料
    valid_y_split = y_train.iloc[valid_index]         # 產生驗證資料標籤
    
   
    random_forest = RandomForestClassifier(random_state=6)
    random_forest.fit(train_x_split, train_y_split)

    train_pred_y = random_forest.predict(train_x_split)       # 確認模型是否訓練成功
    train_acc = accuracy_score(train_y_split,         # 計算訓練資料準確度
                               train_pred_y)
    valid_pred_y = random_forest.predict(valid_x_split)       # 驗證模型是否訓練成功
    valid_acc = accuracy_score(valid_y_split,         # 計算驗證資料準確度
                               valid_pred_y)
    # evaluating the model
    print("Training Accuracy :", random_forest.score(train_x_split, train_y_split))
    print("Testing Accuracy :", random_forest.score(valid_x_split, valid_y_split))
    
    train_acc_list.append(train_acc)
    valid_acc_list.append(valid_acc)

print((
        'average train accuracy: {}\n' +
        '    min train accuracy: {}\n' +
        '    max train accuracy: {}\n' +
        'average valid accuracy: {}\n' +
        '    min valid accuracy: {}\n' +
        '    max valid accuracy: {}').format(
        np.mean(train_acc_list),                          # 輸出平均訓練準確度
        np.min(train_acc_list),                           # 輸出最低訓練準確度
        np.max(train_acc_list),                           # 輸出最高訓練準確度
        np.mean(valid_acc_list),                          # 輸出平均驗證準確度
        np.min(valid_acc_list),                           # 輸出最低驗證準確度
        np.max(valid_acc_list)                            # 輸出最高驗證準確度
    ))

# last_train = last_train.reset_index(drop=True)
# last_test = last_test.reset_index(drop=True)
# test_pred_y = random_forest.predict(last_train)
# # print(last_test)
# # print(last_train)

# count = 0
# for index in range(len(last_test)):
#     if ((test_pred_y[index] - last_test[index]) == 0):
#         count += 1

# print(count / len(last_test))