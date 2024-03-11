from sklearn.cross_decomposition import PLSCanonical
import statistics
import numpy as np
from sklearn.cross_decomposition import PLSCanonical
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.metrics import mean_squared_error
from sklearn.linear_model import LinearRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from matplotlib.pylab import style
style.use('ggplot')    
plt.rcParams['font.sans-serif'] = ['SimHei'] 
plt.rcParams['axes.unicode_minus'] = False 
from matplotlib import rcParams, cycler

#数据导入
df = pd.read_excel("data1.xlsx")
data = np.array(df.values)
#冷却后重量
X_train = data[200:1000,[8,9,10,11,12,13,14,17,25]]
Y_train = data[200:1000,[26]]
X = data[100:1900,[8,9,10,11,12,13,14,17,25]]
X3 = data[100:1900,[25]]
Y_true = data[100:1900,[26]]
#断面扫描数据
X_train2 = data[200:1000,[8,9,10,11,12,13,14,17,24,25,27,28,29,30]]
Y_train2 = data[200:1000,[18,20,21,22,23]]
X2 = data[1000:1500,[8,9,10,11,12,13,14,17,24,25,27,28,29,30]]
Y_true2 = data[1000:1500,[18,20,21,22,23]]

#PLS
scaler = StandardScaler()
#求均值mean = np.mean(X_train,axis=0)
#求标准差std = np.std(X_train,axis=0)
X3 = scaler.fit_transform(X3)
X_train_s = scaler.fit_transform(X_train)#E0
Y_train_s = scaler.fit_transform(Y_train)#F0
E = X_train_s
F = Y_train_s
c = np.array(np.zeros((1,1)))
t = []
r = []
p = []
w = np.array(np.zeros((9, 1)))
for indx in range(3):
    A = E.T@F@F.T@E
    eigenvalues, eigenvectors = np.linalg.eig(A) #q求特征值和特征向量
    max_eigenvalue_index = np.argmax(eigenvalues) #选取最大的特征向量
    w1 = abs(eigenvectors[:, [max_eigenvalue_index]]) #得到w1,abs()用于取实部。因为a+0i
    w = np.append(w,w1,axis=1) #将参数放入w中
    B = F.T@E@E.T@F
    eigenvalues, eigenvectors = np.linalg.eig(B) #q求特征值和特征向量
    max_eigenvalue_index = np.argmax(eigenvalues) #选取最大的特征向量
    c1 = abs(eigenvectors[:, [max_eigenvalue_index]]) #得到w1,abs()用于取实部。因为a+0i
    c = np.append(c,c1,axis=1) #将参数放入c中
    t1 = X_train_s@w1 #计算t1
    t1 = t1.reshape((800, 1)) #将一维数组转化为矩阵
    t.append(t1)
    t_norm = np.linalg.norm(t1) #计算模值
    p1 = E.T@t1/(t_norm*t_norm) #计算p1，r1
    p1 = p1.reshape((9, 1))
    p.append(p1)
    r1 = F.T@t1/(t_norm*t_norm)
    r.append(r1)
    E = E - t1@p1.T #更新E残差
    F = F - t1@r1.T #更新F残差
    if indx == 2:
        F_A = F
w = w[:,1:4] #取参数
c = c[:,1:4] #取参数
#预测模型
X = scaler.fit_transform(X) #数据标准化
Y_true = scaler.fit_transform(Y_true) #数据标准化
#参数计算
t_pre = []
u_pre = []
r_pre = []
for i in range(3):
    t1 = X@w[:,i]
    t_pre.append(t1)
    t1_norm = np.linalg.norm(t1)
    r1 = Y_true.T@t1/(t1_norm*t1_norm)
    r_pre.append(r1)
    u1 = Y_true@c[:,i]
    u_pre.append(u1)
t_pre = np.array(t_pre)
t_pre = t_pre.T
u_pre = np.array(u_pre)
u_pre = u_pre.T
r_pre = np.array(r_pre)
r_pre = r_pre.T
#取参数
t_pre1 = t_pre[:,[0]]
t_pre2= t_pre[:,[1]]
t_pre3 = t_pre[:,[2]]
r_pre1 = r_pre[:,[0]].T
r_pre2 = r_pre[:,[1]].T
r_pre3 = r_pre[:,[2]].T
#F_A = np.array(F_A)
F_pre = t_pre1@r_pre1 + t_pre2@r_pre2 + t_pre3@r_pre3

#结果评估
RMSE1 = mean_squared_error(Y_true,X3,squared=False)
gap = X3-Y_true
mean = np.mean(gap)
std = np.std(gap)
gap_stander = (gap-mean)/std

#预测评分

#绘图
from matplotlib.lines import Line2D
custom_lines = [Line2D([0], [0], c="red", lw=4),
                Line2D([0], [0], c="blue", lw=4)]
fig, ax = plt.subplots()
ax.legend(custom_lines, ['实际称量结果', '预测结果'])
epochs = list(range(1, len(Y_true) + 1))
epoch = list(range(2, len(Y_true) + 2))
plt.xlabel("采样点")
plt.ylabel("后称量质量/g")
plt.plot(epochs,Y_true,c='red',label='实际称量结果')
plt.plot(epoch,X3,c='blue',label='预测结果')
fig2 = plt.figure()
plt.plot(epochs,gap_stander) 
plt.show()
