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
X = data[200:1900,[8,9,10,11,12,13,14,17,25]]
Y_true = data[200:1900,[26]]
x_k = data[0:1000,[8,9,10,11,12,13,14,17,25,26]]
#断面扫描数据
X_train2 = data[200:1000,[8,9,10,11,12,13,14,17,24,25]]
Y_train2 = data[200:1000,[18,20,21,22,23]]
X2 = data[0:1600,[8,9,10,11,12,13,14,17,24,25]]
Y_true2 = data[0:1600,[18,20,21,22,23]]
#PLS2
pls1 = PLSRegression(n_components=3)
pls1.fit(X_train, Y_train)
pls2 = PLSRegression(n_components=5)
pls2.fit(X_train2,Y_train2)
#show pls2.coef_ 
print(np.round(pls1.coef_, 1))
Y_Pre = pls1.predict(X)
Y_Pre2 = pls2.predict(X2)
""" #Pcr
pca_5 = make_pipeline(PCA(n_components=5),LinearRegression())
pcr = pca_5.fit(X_train2,Y_train2)
pre = pcr.predict(X2)
#Ica+pls
ica = FastICA(n_components=5)
X_new = ica.fit_transform(X2)
pls3 = PLSRegression(n_components=3)
pls3.fit(X_new,Y_true2)
X21 = ica.fit_transform(X2)
Pre2 = pls3.predict(X21) """
#结果评估-RMSE
RMSE=[]
RMSE1 = mean_squared_error(Y_true[10:1690],Y_Pre[10:1690],squared=False)
RMSE2 = mean_squared_error(Y_true2[:,0],Y_Pre2[:,0],squared=False)
RMSE3 = mean_squared_error(Y_true2[:,1],Y_Pre2[:,1],squared=False)
RMSE4 = mean_squared_error(Y_true2[:,2],Y_Pre2[:,2],squared=False)
RMSE5 = mean_squared_error(Y_true2[:,3],Y_Pre2[:,3],squared=False)
RMSE6 = mean_squared_error(Y_true2[:,4],Y_Pre2[:,4],squared=False)
RMSE.append(RMSE1.item())
RMSE.append(RMSE2.item())
RMSE.append(RMSE3.item())
RMSE.append(RMSE4.item())
RMSE.append(RMSE5.item())
RMSE.append(RMSE6.item())
print(RMSE)
#预测评分

#print(f"PCR r-squared with 5 components {pca_5.score(X2, Y_true2):.3f}")
#绘图
fig1 = plt.figure(1)
epochs = list(range(1, len(Y_true2) + 1))
plt.plot(epochs,Y_true2[:,[0]],c='red')
plt.plot(epochs,Y_Pre2[:,[0]],c='blue')
fig2 = plt.figure(2)
plt.plot(epochs,Y_true2[:,[1]],c='red')
plt.plot(epochs,Y_Pre2[:,[1]],c='blue')
fig3 = plt.figure(3)
plt.plot(epochs,Y_true2[:,[2]],c='red')
plt.plot(epochs,Y_Pre2[:,[2]],c='blue')
fig4 = plt.figure(4)
plt.plot(epochs,Y_true2[:,[3]],c='red')
plt.plot(epochs,Y_Pre2[:,[3]],c='blue')
fig5 = plt.figure(5)
plt.plot(epochs,Y_true2[:,[4]],c='red')
plt.plot(epochs,Y_Pre2[:,[4]],c='blue')
fig6 = plt.figure(6)
epochs = list(range(1, len(Y_true) + 1))
epoch = list(range(2, len(Y_true) + 2))
from matplotlib.lines import Line2D
custom_lines = [Line2D([0], [0], c="red", lw=4),
                Line2D([0], [0], c="blue", lw=4)]
fig, ax = plt.subplots()
ax.legend(custom_lines, ['实际称量结果', '预测结果'])
epochs = list(range(1, len(Y_true[10:1690,[0]]) + 1))
epoch = list(range(1, len(Y_true[10:1690,[0]]) + 1))
plt.xlabel("采样点")
plt.ylabel("后称量质量/g")
plt.plot(epochs,Y_true[10:1690,[0]],c='red',label='实际称量结果')
plt.plot(epoch,Y_Pre[10:1690,[0]],c='blue',label='预测结果')
plt.show()
