from sklearn.cross_decomposition import PLSCanonical
import numpy as np
from sklearn.cross_decomposition import PLSCanonical
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.metrics import mean_squared_error
from sklearn.linear_model import LinearRegression
from sklearn.pipeline import make_pipeline
from sklearn.decomposition import PCA
from sklearn.decomposition import FastICA
from sklearn.cross_decomposition import PLSRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics.pairwise import rbf_kernel
from matplotlib.pylab import style
style.use('ggplot')    
plt.rcParams['font.sans-serif'] = ['SimHei'] 
plt.rcParams['axes.unicode_minus'] = False 
#数据导入
df = pd.read_excel("data1.xlsx")
data = np.array(df.values)
#冷却后重量
X_train = data[200:1900,[8,9,10,11,12,13,14,17,25]]
Y_train = data[200:1900,[26]]
X = data[200:1900,[8,9,10,11,12,13,14,17,25]]
Y_true = data[200:1900,[26]]
x_stander =X_train[[0],]
#断面扫描数据
X_train2 = data[200:1000,[8,9,10,11,12,13,14,17,24,25,27,28,29,30]]
Y_train2 = data[200:1000,[18,20,21,22,23]]
X2 = data[1000:1500,[8,9,10,11,12,13,14,17,24,25,27,28,29,30]]
Y_true2 = data[1000:1500,[18,20,21,22,23]]
#PLS
scaler = StandardScaler()
#求均值mean = np.mean(X_train,axis=0)
#求标准差std = np.std(X_train,axis=0)

# 计算RBF核
rbf_X = rbf_kernel(X_train,x_stander, gamma=0.1)  # gamma是RBF核的带宽参数
X_train = np.append(X_train,rbf_X,axis=1)
rbf_X = rbf_kernel(X,x_stander, gamma=0.1)
X = np.append(X,rbf_X,axis=1)
""" rbf_Y = rbf_kernel(Y_train, gamma=0.1)
rbf_X1 = rbf_kernel(X, gamma=0.1)
rbf_Y1 = rbf_kernel(Y_true,gamma=0.1) """
#pls
pls1 = PLSRegression(n_components=3)
pls1.fit(X_train, Y_train)
print(np.round(pls1.coef_, 1))
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
Y_train = scaler.fit_transform(Y_train)
#结果测试
y_pre = pls1.predict(X)
RMSE1 = mean_squared_error(Y_true[10:1900],y_pre[10:1900],squared=False)
from matplotlib.lines import Line2D
custom_lines = [Line2D([0], [0], c="red", lw=4),
                Line2D([0], [0], c="blue", lw=4)]
fig, ax = plt.subplots()
ax.legend(custom_lines, ['后称量标准化结果', '前称量标准化结果'])
plt.xlabel("采样点")
plt.ylabel("称量质量/g")
epochs = list(range(1, len(X_train) + 1))
plt.plot(epochs,X_train[:,[8]],c='blue')
plt.plot(epochs,Y_train,c='red')
plt.show()
print(RMSE1)