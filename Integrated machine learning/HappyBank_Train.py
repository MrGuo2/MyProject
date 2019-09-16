#%%
# 导入pandas相关的工具包
import numpy as np 
import pandas as pd 
import matplotlib as plt

dpath = "./MachineLearning/HappyBank/"
Train = pd.read_csv(dpath+"Train.csv")
Test = pd.read_csv(dpath+"Test.csv")

#%%
# 剔除Train和Test缺失值
Train = Train.drop(["LoggedIn", "Interest_Rate", "Processing_Fee", 
"EMI_Loan_Submitted","Loan_Amount_Submitted", "Loan_Tenure_Submitted"], axis=1)
Test = Test.drop(["Interest_Rate", "Processing_Fee", "EMI_Loan_Submitted",
"Loan_Amount_Submitted", "Loan_Tenure_Submitted"], axis=1)

#%%
# 删除数量较少的缺失值
willDrop = ["City", "Loan_Amount_Applied", "Loan_Tenure_Applied", 
"Employer_Name", "Var1"]
Train = Train.dropna(subset = willDrop)
Test = Test.dropna(subset = willDrop)

#%%
# 填充Salary_Account中的null值
Train['Salary_Account'] = Train['Salary_Account'].fillna('unknown')
Test["Salary_Account"] = Test["Salary_Account"].fillna("unknown")
X_train = Train.drop(["Disbursed"], axis = 1)
y_train = Train["Disbursed"]

#%% [markdown]
# # 训练数据

#%%
# 导入LabelEncoder和LGBM的工具包
import lightgbm as lgbm 
from lightgbm.sklearn import LGBMClassifier
from sklearn.model_selection import GridSearchCV
import matplotlib.pyplot as plt

#%%
# 将所有参数转化为数值
# 将test拼接到train里，然后统一labelencoder
from sklearn.preprocessing import LabelEncoder
LE = LabelEncoder()

willLE = ["ID", "Gender", "City", "DOB", "Lead_Creation_Date", "Employer_Name", "Var2", 
"Salary_Account", "Mobile_Verified", "Var1", "Var5", "Filled_Form", "Device_Type", "Source"]
for col in willLE:
    colLabEn = LE.fit(X_train[col].append(Test[col]).astype(str))
    X_train[col] = colLabEn.transform(X_train[col].astype(str))
    Test[col] = colLabEn.transform(Test[col].astype(str))
print("Transform successful!")
#%%
# 将所有数据归一化处理
# X_train_cols = X_train.columns
# Test_cols = Test.columns

# from sklearn.preprocessing import MinMaxScaler
# ms = MinMaxScaler()

# X_train = ms.fit_transform(X_train)
# Test = ms.fit_transform(Test)
# X_train = pd.DataFrame(columns = X_train_cols, data = X_train)
# Test = pd.DataFrame(columns = Test_cols, data = Test)
#%% [markdown]
# # LightGBM参数调优
# LightGBM的主要的超参数包括  
# 1. 树的数目n_estimators和学习率learning_rate
# 2. 树的最大深度max_depth和树的最大叶子节点数目num_leaves(注意:XGboot只有max_depth,
# LightGBM采用叶子优先的方式生成树，num_leaves很重要，设置成比2^max_depth小)
# 3. 叶子节点的最小样本数:min_data_in_leaf(min_data, min_child_samples)
# 4. 每颗数的列采样比例:feature_fraction/colsample_bytree
# 5. 每棵树的行采样比例:bagging_fraction(需要同时设置bagging_freq=1)/subsample
# 6. 正则化参数lambda_l1(reg_alpha),lambda_l2(reg_lambda)
# 7. 两个模型复杂度参数，但会影响模型速度和精度。可根据特征取值范围和样本数目修改这两个参数
# 1) 特征的最大bin数目max_bin:默认255
# 2) 用来建立直方图的样本数目subsample_for_bin:默认200000  
# 对n_estimators，用LightGBM内嵌的cv函数调优，因为同XGBoost一样，LightGBM学习过程内嵌了
# cv，速度极快。对其他参数用GridSearchCV

#%% [markdown]
# # nestimators

#%%
# 计算参数对应的得分并求出初步的nestimators
MAX_ROUNDS = 10000
def get_n_estimators(params, X_train, y_train, early_stopping_rounds=10):
    lgbm_param = params.copy()
    lgbmtrain = lgbm.Dataset(X_train, y_train)
    cv_result = lgbm.cv(lgbm_param, lgbmtrain, num_boost_round=MAX_ROUNDS, nfold=3, metrics="binary_logloss", 
    early_stopping_rounds=early_stopping_rounds, seed=3)
    print("best n_estimators:", len(cv_result["binary_logloss-mean"]))
    print("best cv score:", cv_result["binary_logloss-mean"][-1])
    
    return len(cv_result["binary_logloss-mean"])

#%%
# 输入参数
params = {"boosting_type":"gbdt", "objective":"binary", "n_jobs":8, "learning_rate":0.1, 
"num_leaves":60, "max_depth":6, "max_bin":127, "subsample":0.7, "bagging_freq":1, "colsample_bytree":0.7}

n_estimators_1 = get_n_estimators(params, X_train, y_train)

#%%
# num_leaves&max_depth = 7
# num_leaves建10~60，值越大模型越复杂，越容易过拟合响应的扩大max_depth=7
from sklearn.model_selection import StratifiedKFold
kfold = StratifiedKFold(n_splits=3, shuffle=True, random_state=3)

params = {"boosting_type":"gbdt", "objective":"binary",
"n_jobs":8, "learning_rate":0.1, "n_estimators":n_estimators_1, "max_depth":7,
"max_bin":127, "subsample":0.7, "bagging_freq":1, "colsample_bytree":0.7}
# silent是否在boosting过程中输出
lg = LGBMClassifier(silent=False, **params)

num_leaves_s = range(10, 60, 10)
tuned_parameters = dict(num_leaves = num_leaves_s)

grid_search = GridSearchCV(lg, n_jobs=4, param_grid=tuned_parameters, cv=kfold, 
scoring="neg_log_loss", verbose=5, refit=False)
grid_search.fit(X_train, y_train)

#%%
# 显示最终合适得num_leaves(30)
print(-grid_search.best_score_)
print(grid_search.best_params_)
test_means = grid_search.cv_results_["mean_test_score"]
test_stds = grid_search.cv_results_["std_test_score"]
train_means = grid_search.cv_results_["mean_train_score"]
train_stds = grid_search.cv_results_["std_train_score"]

n_leafs = len(num_leaves_s)
X_axis = num_leaves_s
plt.plot(X_axis, -test_means)

plt.xlabel("num_leaves")
plt.ylabel("Log Loss")
plt.show()
#%% [markdown]
# 叶子节点的最小样本数目  
# 叶子节点数目:30,共2类，平均每类40个叶子节点  
# 每棵树的样本数目最少的类(稀有事件)的样本数目:5000(最小样本数目)*2/3(3折交叉验证)*0.7(subsample)=2333
# 所以每个叶子节点2300/30 = 70个样本点
# 搜索范围 10~100

#%%
# 调整min_child_sample的参数(100)
params = {"boosting_type":"gbdt", "objective":"binary", "num_leaves":30, 
        "n_jobs":8, "learning_rate":0.1, "n_estimators":n_estimators_1, "max_depth":7,
        "max_bin":127, "subsample":0.7, "bagging_freq":1, "colsample_bytree":0.7}

lg = LGBMClassifier(silent=False, **params)
min_child_samples = range(60, 130, 10)
tuned_parameters = dict(min_child_samples=min_child_samples)

grid_search = GridSearchCV(lg, n_jobs=8, param_grid=tuned_parameters, cv = kfold,
                           scoring='neg_log_loss', verbose=5, refit=False)
grid_search.fit(X_train, y_train)  

#%%
# 显示min_child_sample cv误差曲线
print(-grid_search.best_score_)
print(grid_search.best_params_)

test_means = grid_search.cv_results_["mean_test_score"]
test_stds = grid_search.cv_results_["std_test_score"]
train_means = grid_search.cv_results_["mean_train_score"]
train_stds = grid_search.cv_results_["std_train_score"]

X_axis = min_child_samples
plt.plot(X_axis, -test_means)
plt.show()
#%%
# 调试参数sub_sample/bagging_fraction(0.7)
params = {"boosting_type":"gbdt", "objective":"binary", "num_leaves":30, 
        "n_jobs":8, "learning_rate":0.1, "n_estimators":n_estimators_1, "max_depth":7,
        "max_bin":127, "min_child_samples":100, "bagging_freq":1, "colsample_bytree":0.7}

lg = LGBMClassifier(silent=False, **params)
subsample_s = [i/10.0 for i in range(5,10)]
tuned_parameters = dict(subsample = subsample_s)

grid_search = GridSearchCV(lg, n_jobs=8, param_grid=tuned_parameters, cv=kfold,
                        scoring="neg_log_loss", verbose=5, refit=False)
grid_search.fit(X_train, y_train)
#%%
# #显示subsample cv的误差曲线
print(-grid_search.best_score_)
print(grid_search.best_params_)

test_means = grid_search.cv_results_["mean_test_score"]
test_stds = grid_search.cv_results_["std_test_score"]
train_means = grid_search.cv_results_["mean_train_score"]
train_stds = grid_search.cv_results_["std_train_score"]

X_axis = subsample_s
plt.plot(X_axis, -test_means)
plt.show()
#%%
# 调试参数sub_feature/feature_fraction/colsample_bytree(0.7)
params = {"boosting_type":"gbdt", "objective":"binary", "num_leaves":30, 
        "n_jobs":8, "learning_rate":0.1, "n_estimators":n_estimators_1, "max_depth":7,
        "max_bin":127, "min_child_samples":100, "bagging_freq":1, "subsample":0.7}

lg = LGBMClassifier(silent=False, **params)
colsample_bytree = [i/10.0 for i in range(5, 10)]
tuned_parameters = dict(colsample_bytree=colsample_bytree)

grid_search = GridSearchCV(lg, n_jobs=8, param_grid=tuned_parameters, cv=kfold,
                        scoring="neg_log_loss", verbose=5, refit=False)
grid_search.fit(X_train, y_train)
#%%
print(-grid_search.best_score_)
print(grid_search.best_params_)

test_means = grid_search.cv_results_[ 'mean_test_score' ]
test_stds = grid_search.cv_results_[ 'std_test_score' ]
train_means = grid_search.cv_results_[ 'mean_train_score' ]
train_stds = grid_search.cv_results_[ 'std_train_score' ]

x_axis = colsample_bytree
plt.plot(x_axis, -test_means)
plt.show()

#%%
# 减少学习率，调整n_estimators
params = {"boosting_type":"gbdt", "objective":"binary", "num_leaves":30, 
        "n_jobs":8, "learning_rate":0.01,  "max_depth":7, #'n_estimators':n_estimators_1,
        "max_bin":127, "min_child_samples":100, "bagging_freq":1, "subsample":0.7}

n_estimators_2 = get_n_estimators(params , X_train , y_train)
#%%
params = {"boosting_type":"gbdt", "objective":"binary", "num_leaves":30, 
        "n_jobs":8, "learning_rate":0.01,  "max_depth":7, 'n_estimators':n_estimators_2,
        "max_bin":127, "min_child_samples":100, "bagging_freq":1, "subsample":0.7}

lg = LGBMClassifier(silent=False,  **params)
lg.fit(X_train, y_train)

#%%
# 存储特征
import pickle
pickle.dump(lg, open(dpath+"HappyBank_LGBM.pkl", "wb"))
pickle.dump(LE, open(dpath+"HappyBank_LabEn.pkl", "wb"))
#%%
# 输出特征重要性
# 使用pd.Series进行组合，值是特征重要性的值，index是样本特征，.sort_value 进行排序操作
feature_important = pd.Series(lg.feature_importances_, index = X_train.columns).sort_values(ascending=False)
plt.bar(feature_important.index, feature_important.data)
plt.figure(figsize=(13, 6))
plt.show()
#%%
