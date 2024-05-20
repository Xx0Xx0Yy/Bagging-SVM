import loaddata as lddata
from sklearn.cluster import DBSCAN
from sklearn import svm
from sklearn.model_selection import train_test_split
import sklearn.metrics as sm
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import BaggingClassifier
from sklearn.model_selection import GridSearchCV
from imblearn.over_sampling import SMOTE, BorderlineSMOTE
from imblearn.over_sampling import RandomOverSampler
from imblearn.under_sampling import RandomUnderSampler, NearMiss
from imblearn.pipeline import Pipeline
import numpy as np
import pandas as pd
from sklearn.ensemble import AdaBoostClassifier
from sklearn.feature_selection import RFE
from sklearn.decomposition import PCA
from sklearn.feature_selection import VarianceThreshold
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import VarianceThreshold
from sklearn.metrics import accuracy_score
import xlwt
import time

time_start = time.time()

x = lddata.clean_x

# 加载对比实验数据
dataset = [[0 for j in range(4)] for h in range(len(x))]
for j in range(len(x)):
    dataset[j][0] = lddata.clean_speed[j]
    dataset[j][1] = lddata.angular_diff[j]
    dataset[j][2] = lddata.clean_x[j]
    dataset[j][3] = lddata.clean_y[j]

# 标签
label = lddata.newrow_tag

x_train_small, x_test_small, y_train_small, y_test_small = train_test_split(dataset, label, test_size=0.3)


# 设置Bagging-SVM数据格式
dataSet = [[0 for j in range(43)] for h in range(len(x))]
for j in range(len(x)):
    dataSet[j][0] = lddata.clean_speed[j]
    dataSet[j][1] = lddata.acc[j]
    dataSet[j][2] = lddata.angular_diff[j]

    dataSet[j][3] = lddata.angular_speed[j]
    dataSet[j][4] = lddata.angular_acclec[j]
    # dataSet[j][5] = lddata.clean_x[j]
    # dataSet[j][6] = lddata.clean_y[j]

    dataSet[j][5] = lddata.distance[j]

    dataSet[j][6] = lddata.acc_SD[j]
    dataSet[j][7] = lddata.angular_acclec_SD[j]
    dataSet[j][8] = lddata.angular_diff_SD[j]
    dataSet[j][9] = lddata.angular_speed_SD[j]
    dataSet[j][10] = lddata.speed_SD[j]

    dataSet[j][11] = lddata.acc_med[j]
    dataSet[j][12] = lddata.angular_acclec_med[j]
    dataSet[j][13] = lddata.angular_diff_med[j]
    dataSet[j][14] = lddata.angular_speed_med[j]
    dataSet[j][15] = lddata.speed_med[j]

    dataSet[j][16] = lddata.angular_acclec_max[j]
    dataSet[j][17] = lddata.speed_max[j]
    dataSet[j][18] = lddata.angular_diff_max[j]
    dataSet[j][19] = lddata.angular_speed_max[j]
    dataSet[j][20] = lddata.acc_max[j]

    dataSet[j][21] = lddata.angular_acclec_min[j]
    dataSet[j][22] = lddata.speed_min[j]
    dataSet[j][23] = lddata.angular_diff_min[j]
    dataSet[j][24] = lddata.angular_speed_min[j]
    dataSet[j][25] = lddata.acc_min[j]

    dataSet[j][26] = lddata.speed_d[j]
    dataSet[j][27] = lddata.acc_d[j]
    dataSet[j][28] = lddata.angular_diff_d[j]
    dataSet[j][29] = lddata.angular_speed_d[j]
    dataSet[j][30] = lddata.angular_acclec_d[j]

    dataSet[j][31] = lddata.angular_acclec_e[j]
    dataSet[j][32] = lddata.speed_e[j]
    dataSet[j][33] = lddata.angular_diff_e[j]
    dataSet[j][34] = lddata.angular_speed_e[j]
    dataSet[j][35] = lddata.acc_e[j]


    dataSet[j][36] = lddata.acc_AVER[j]
    dataSet[j][37] = lddata.angular_acclec_AVER[j]
    dataSet[j][38] = lddata.angular_diff_AVER[j]
    dataSet[j][39] = lddata.angular_speed_AVER[j]
    dataSet[j][40] = lddata.speed_AVER[j]

    dataSet[j][41] = lddata.clean_x[j]
    dataSet[j][42] = lddata.clean_y[j]


    # dataSet[j][1] = lddata.acc_SD[j]
    # dataSet[j][2] = lddata.angular_acclec_SD[j]
    # dataSet[j][3] = lddata.angular_diff_SD[j]
    # dataSet[j][4] = lddata.angular_speed_SD[j]
    # dataSet[j][5] = lddata.speed_SD[j]
    #
    # dataSet[j][6] = lddata.acc_med[j]
    # dataSet[j][7] = lddata.angular_acclec_med[j]
    # dataSet[j][8] = lddata.angular_diff_med[j]
    # dataSet[j][9] = lddata.angular_speed_med[j]
    # dataSet[j][10] = lddata.speed_med[j]
    #
    # dataSet[j][11] = lddata.angular_acclec_max[j]
    # dataSet[j][12] = lddata.speed_max[j]
    # dataSet[j][13] = lddata.angular_diff_max[j]
    # dataSet[j][14] = lddata.angular_speed_max[j]
    # dataSet[j][15] = lddata.acc_max[j]
    #
    # dataSet[j][16] = lddata.angular_acclec_min[j]
    # dataSet[j][17] = lddata.speed_min[j]
    # dataSet[j][18] = lddata.angular_diff_min[j]
    # dataSet[j][19] = lddata.angular_speed_min[j]
    # dataSet[j][20] = lddata.acc_min[j]
    #
    # dataSet[j][21] = lddata.speed_d[j]
    # dataSet[j][22] = lddata.acc_d[j]
    # dataSet[j][23] = lddata.angular_diff_d[j]
    # dataSet[j][24] = lddata.angular_speed_d[j]
    # dataSet[j][25] = lddata.angular_acclec_d[j]
    #
    # dataSet[j][26] = lddata.angular_acclec_e[j]
    # dataSet[j][27] = lddata.speed_e[j]
    # dataSet[j][28] = lddata.angular_diff_e[j]
    # dataSet[j][29] = lddata.angular_speed_e[j]
    # dataSet[j][30] = lddata.acc_e[j]
    #
    #
    # dataSet[j][31] = lddata.acc_AVER[j]
    # dataSet[j][32] = lddata.angular_acclec_AVER[j]
    # dataSet[j][33] = lddata.angular_diff_AVER[j]
    # dataSet[j][34] = lddata.angular_speed_AVER[j]
    # dataSet[j][35] = lddata.speed_AVER[j]

# dataSet=lddata.dataSet
# label=lddata.newrow_tag
# dataSet = pd.DataFrame(dataSet)
# writer = pd.ExcelWriter('paddy_1_1.xlsx')		# 写入Excel文件
# dataSet.to_excel(writer, 'paddy', float_format='%.5f')		# ‘page_1’是写入excel的sheet名
# writer.save()
# writer.close()

# book = xlwt.Workbook(encoding='utf-8',style_compression=0)
# sheet = book.add_sheet('wheat',cell_overwrite_ok=True)
# for i in range(len(x)):
#     data=dataSet[i]
#     for j in range(43):
#         sheet.write(i+1,j,data[j])

# savepath = 'wheat_2_1.xlsx'
# book.save(savepath)

# 采样
# 过采样
borderline_smote = BorderlineSMOTE(k_neighbors=5, kind="borderline-1")
smote = SMOTE(k_neighbors=5, random_state=42)
ros = RandomOverSampler(random_state=0, sampling_strategy='auto')
# 欠采样
nearmiss = NearMiss(version=3)
random_undersampler = RandomUnderSampler()
# 组合采样
steps_borderline = [('SMOTE', borderline_smote),
                    ('Rand_Undersampler', nearmiss)]


pipe_borderline = Pipeline(steps=steps_borderline)
sample_dataSet,sample_label = pipe_borderline.fit_resample(dataSet,label)

# PCA
dataSet = StandardScaler().fit_transform(sample_dataSet)
label = sample_label
model_lda = PCA(n_components=0.99)
model_lda.fit(dataSet, label)
dataSet = model_lda.transform(dataSet)

# dbscan聚类方式 clustering为聚类结果
# clustering = DBSCAN(eps=parm.eps, min_samples=parm.min_samples).fit(dataSet)

x_train, x_test, y_train, y_test = train_test_split(dataSet, label, test_size=0.3)


# bagging-SVM
clf = svm.SVC(kernel='rbf')
model = BaggingClassifier(base_estimator=clf, n_estimators=10)
model.fit(x_train, y_train)
y_pred1 = model.predict(x_test)

# param_grid = [{
#                'n_estimators': [1, 5, 10, 20],
#                'max_samples': [0.5, 0.75, 1.0],
#                'bootstrap':[True, False]}]
#
# grid = GridSearchCV(model, param_grid, verbose=0, cv=10)
# grid.fit(x_train, y_train)
# best_parameters = grid.best_estimator_.get_params()
# for para, val in list(best_parameters.items()):
#     print(para, val)
# best_estimator = grid.best_estimator_
# y_pred1 = best_estimator.predict(x_test)


# DecisionTree
# dt = DecisionTreeClassifier(criterion="entropy")
# dt = dt.fit(x_train_small,y_train_small)
# y_pred2 = dt.predict(x_test_small)

# # RandomForest
# rfc = RandomForestClassifier(random_state=1)
# rfc = rfc.fit(x_train_small, y_train_small)
# y_pred3 = rfc.predict(x_test_small)

# SVM
# clf = svm.SVC(kernel='rbf')
# param_grid = {'C': [0.1,1,3,5,8,10,10.5,11,11.5,12],
#               'gamma': [0.001,0.005,0.01,0.015,0.1]}
# grid_search = GridSearchCV(clf, param_grid, verbose=0, cv=5)
# grid_search.fit(x_train, y_train)
# grid_search.fit(x_train, y_train)
# best_parameters = grid_search.best_estimator_.get_params()
# for para, val in list(best_parameters.items()):
#     print(para, val)
# clf = svm.SVC(kernel='rbf', C=best_parameters['C'], gamma=best_parameters['gamma'])
# clf.fit(x_train, y_train)

# y_pred1 = clf.predict(x_test)
# y_pred1_label = pd.DataFrame(y_pred1)
# writer = pd.ExcelWriter('paddy_3_label.xlsx')
# y_pred1_label.to_excel(writer, 'paddy_label_feature', float_format='%.5f')		# ‘page_1’是写入excel的sheet名
# writer.save()
# writer.close()

# Adaboost
# model = AdaBoostClassifier(algorithm="SAMME", n_estimators=2, base_estimator=clf)
# model.fit(over_sample_x, over_sample_y)
# y_pred1 = model.predict(x_test)



time_end = time.time()
print('\ntimecost:',time_end-time_start,'s')

result1 = sm.classification_report(y_test, y_pred1)
print(result1)
print(accuracy_score(y_test, y_pred1))



