
import scipy.stats
from sklearn import linear_model
import pandas as pd
from math import radians, cos, sin, asin, sqrt
import matplotlib.pyplot as plt
from sklearn.cluster import DBSCAN
import numpy as np
import math
import random
from sklearn.metrics import precision_score, recall_score, f1_score
import multiprocessing
from time import time
from datetime import datetime, date
import heapq
from getdistance import haversine
from scipy import stats

print('loading data...\n')
#加载清理好的数据
#data1作为pands读取数据的中间转换
data1 = pd.read_excel("agricultural machinery/data/clean_wheat_1.xlsx", header=0)
cleandata = data1.loc[:,['经度', '纬度', '时间', '速度', '方向', '高度', '标签']]


#根据算法需要读取清洗好数据的属性
clean_x = cleandata['经度']
clean_y = cleandata['纬度']
clean_speed = cleandata['速度']
newrow_tag = cleandata['标签']
direct = cleandata['方向']
# clean_x = cleandata['LNG']
# clean_y = cleandata['LAT']
# clean_speed = cleandata['SPEED']
# clean_id = cleandata['ID']
# direct = cleandata['DIRECTION']
# clean_x = cleandata['longitude']
# clean_y = cleandata['latitude']
# clean_speed = cleandata['speed']
# newrow_tag = cleandata['tags']
# direct = cleandata['dir']
# clean_id = cleandata['latitude']


def SD(num,lst):
    SD=[0 for x in range(len(lst))]
    SD[0]=np.std([lst[0]])
    for i in range(1,len(lst)):
        if(i<=num):
            SD[i]=np.std(lst[:i+1])
        else:
            SD[i] = np.std(lst[i-num:i+1])
    return SD

def AVER(num,lst):
    AVER=[0 for x in range(len(lst))]
    AVER[0]=np.mean([lst[0]])
    for i in range(1,len(lst)):
        if(i<=num):
            AVER[i]=np.mean(lst[:i+1])
        else:
            AVER[i] = np.mean(lst[i-num:i+1])
    return AVER

def get_median(data):
   data = sorted(data)
   size = len(data)
   if size % 2 == 0: # 判断列表长度为偶数
    #median = (data[size//2]+data[size//2-1])/2
    median =  data[size//2]
    data[0] = median
   if size % 2 == 1: # 判断列表长度为奇数
    median = data[(size-1)//2]
    data[0] = median
   return data[0]

def med(num,lst):
    med=[0 for x in range(len(lst))]
    med[0]=lst[0]
    for i in range(1,len(lst)):
        if(i<=num):
            med[i]=get_median(lst[:i+1])
        else:
            med[i] = get_median(lst[i-num:i+1])
    return med

def findMAX(num,lst):
    findmax = [0 for x in range(len(lst))]
    findmax[0] = lst[0]
    for i in range(1, len(lst)):
        if (i <= num):
            findmax[i] = max(lst[:i + 1])
        else:
            findmax[i] = max(lst[i - num:i + 1])
    return findmax

def findMIN(num,lst):
    findmin = [0 for x in range(len(lst))]
    findmin[0] = lst[0]
    for i in range(1, len(lst)):
        if (i <= num):
            findmin[i] = max(lst[:i + 1])
        else:
            findmin[i] = max(lst[i - num:i + 1])
    return findmin

def Skew(num,lst):
    Skew = [0 for x in range(len(lst))]
    Skew[0] = lst[0]
    for i in range(1, len(lst)):
        if (i <= num):
            Skew[i] = stats.skew(lst[:i + 1], bias=False)
        else:
            Skew[i] = stats.skew(lst[i - num:i + 1], bias=False)
    return Skew

def Kurt(num,lst):
    Kurt = [0 for x in range(len(lst))]
    Kurt[0] = lst[0]
    for i in range(1, len(lst)):
        if (i <= num):
            Kurt[i] = stats.kurtosis(lst[:i + 1], bias=False)
        else:
            Kurt[i] = stats.kurtosis(lst[i - num:i + 1], bias=False)
    return Kurt

def CV(num,lst):
    CV = [0 for x in range(len(lst))]
    CV[0] = lst[0]
    for i in range(1, len(lst)):
        if (i <= num):
            CV[i] = scipy.stats.variation(lst[:i+1], axis=0)
        else:
            CV[i] = scipy.stats.variation(lst[i-num:i+1], axis=0)
    CV = np.array(CV)
    CV = np.nan_to_num(CV)
    CV = CV.tolist()
    return CV


## n个临近点距离
distance=[i for i in range(len(clean_x))]
for i in range(0,len(clean_x)-10):
    dis = 0
    for j in range(i+1,i+11):
        temp=haversine(clean_x[i],clean_y[i],clean_x[j],clean_y[j])
        dis=dis+temp
    distance[i]=dis
for i in range(len(clean_x)-10,len(clean_x)):
    dis = 0
    for j in range(i-1,i-11,-1):
        temp=haversine(clean_x[i],clean_y[i],clean_x[j],clean_y[j])
        dis = dis + temp
    distance[i]=dis

## n个临近点平均距离
ave_distance=[i for i in range(len(clean_x))]
for i in range(0,len(clean_x)-10):
    dis = 0
    for j in range(i+1,i+11):
        temp=haversine(clean_x[i],clean_y[i],clean_x[j],clean_y[j])
        dis=dis+temp
    aver_dis=dis/10
    ave_distance[i]=aver_dis
for i in range(len(clean_x)-10,len(clean_x)):
    dis = 0
    for j in range(i-1,i-11,-1):
        temp=haversine(clean_x[i],clean_y[i],clean_x[j],clean_y[j])
        dis = dis + temp
    aver_dis=dis/10
    ave_distance[i]=aver_dis

dist=[i for i in range(len(clean_x))]
for j in range(len(list(clean_speed))):
    if j == 0:
        dist[j] = 0
    else:
        dist[j] = haversine(clean_x[j],clean_y[j],clean_x[j-1],clean_y[j-1])

# # # 曲率
# S=[i for i in range(len(clean_x))]
# for j in range(len(list(clean_speed))):
#     if j == 0:
#         S[j] = 1
#     else:
#         S[j] = (dist[j]+dist[j+1])/


# #加速度
acc=[i for i in range(len(clean_speed))]
speed_diff=[i for i in range(len(clean_speed))]
for j in range(len(list(clean_speed))):
    if j == 0:
        speed_diff[j] = 0
        acc[j] = 0
    else:
        speed_diff[j] = clean_speed[j] - clean_speed[j - 1]
        acc[j] = float(speed_diff[j]) / 1

# #角度差
angular_diff=[i for i in range(len(direct))]
for l in range(len(list(clean_speed))):
    if l == 0:
        angular_diff[l] = 0
    else:
        angular_diff[l] = direct[l] - direct[l - 1]

# #角速度
angular_speed=[i for i in range(len(direct))]
for temp in range(len(list(clean_speed))):
    if temp == 0:
        angular_speed[temp] = 0
    else:
        if clean_speed[temp] == 0:
            angular_speed[temp] = 0
        else:
            angular_speed[temp] = direct[temp] / 1

# #角加速度
angular_acclec=[i for i in range(len(direct))]
angular_speed_diff=[i for i in range(len(direct))]
for k in range(len(list(clean_speed))):
    if k == 0:
        angular_speed_diff[k] = 0
        angular_acclec[k] = 0
    else:
        angular_speed_diff[k] = angular_speed[k] - angular_speed[k - 1]
        angular_acclec[k] = angular_speed_diff[k] / 1


cit1 = 100
cit2 = 20
# 标准差
speed_SD = SD(cit1, clean_speed)
acc_SD = SD(cit1, acc)
angular_diff_SD = SD(cit1, angular_diff)
angular_speed_SD = SD(cit1, angular_speed)
angular_acclec_SD = SD(cit1, angular_acclec)
# 均值
speed_AVER = AVER(cit1, clean_speed)
acc_AVER = AVER(cit1, acc)
angular_diff_AVER = AVER(cit1, angular_diff)
angular_speed_AVER = AVER(cit1, angular_speed)
angular_acclec_AVER = AVER(cit1, angular_acclec)
# 中位数
speed_med = med(cit1, clean_speed)
acc_med = med(cit1, acc)
angular_diff_med = med(cit1, angular_diff)
angular_speed_med = med(cit1, angular_speed)
angular_acclec_med = med(cit1, angular_acclec)
# 最大值
speed_max = findMAX(cit1, clean_speed)
acc_max = findMAX(cit1, acc)
angular_diff_max = findMAX(cit1, angular_diff)
angular_speed_max = findMAX(cit1, angular_speed)
angular_acclec_max = findMAX(cit1, angular_acclec)
# 最小值
speed_min = findMIN(cit1, clean_speed)
acc_min = findMIN(cit1, acc)
angular_diff_min = findMIN(cit1, angular_diff)
angular_speed_min = findMIN(cit1, angular_speed)
angular_acclec_min = findMIN(cit1, angular_acclec)
# 变异系数
speed_cv = CV(cit1, clean_speed)
acc_cv = CV(cit1, acc)
angular_diff_cv = CV(cit1, angular_diff)
angular_speed_cv = CV(cit1, angular_speed)
angular_acclec_cv = CV(cit1, angular_acclec)
# 偏度
speed_d = Skew(cit1, clean_speed)
acc_d = Skew(cit1, acc)
angular_diff_d = Skew(cit1, angular_diff)
angular_speed_d = Skew(cit1, angular_speed)
angular_acclec_d = Skew(cit1, angular_acclec)
# 峰度
speed_e = Kurt(cit1, clean_speed)
acc_e = Kurt(cit1, acc)
angular_diff_e = Kurt(cit1, angular_diff)
angular_speed_e = Kurt(cit1, angular_speed)
angular_acclec_e = Kurt(cit1, angular_acclec)


