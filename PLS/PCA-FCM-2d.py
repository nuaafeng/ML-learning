import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from sklearn.decomposition import PCA
from sklearn.cross_decomposition import PLSRegression
from sklearn.metrics import mean_squared_error
from sklearn.metrics.pairwise import rbf_kernel
from matplotlib.pylab import style
style.use('ggplot')    
plt.rcParams['font.sans-serif'] = ['SimHei'] 
plt.rcParams['axes.unicode_minus'] = False 
#FCM调用函数
def FCM(X, c_clusters=8, m=2, eps=50):
    membership_mat = np.random.random((len(X), c_clusters))
    membership_mat = np.divide(membership_mat, np.sum(membership_mat, axis=1)[:, np.newaxis])

    while True:
        working_membership_mat = membership_mat ** m
        Centroids = np.divide(np.dot(working_membership_mat.T, X), np.sum(working_membership_mat.T, axis=1)[:, np.newaxis])
        
        n_c_distance_mat = np.zeros((len(X), c_clusters))
        for i, x in enumerate(X):
            for j, c in enumerate(Centroids):
                n_c_distance_mat[i][j] = np.linalg.norm(x-c, 2)
        
        new_membership_mat = np.zeros((len(X), c_clusters))
        
        for i, x in enumerate(X):
            for j, c in enumerate(Centroids):
                new_membership_mat[i][j] = 1. / np.sum((n_c_distance_mat[i][j] / n_c_distance_mat[i]) ** (2 / (m - 1)))
        if np.sum(abs(new_membership_mat - membership_mat)) < eps:
            break
        membership_mat =  new_membership_mat
    return np.argmax(new_membership_mat, axis=1)

#FCM结果评估函数
def evaluate(y, t):
    a, b, c, d = [0 for i in range(4)]
    for i in range(len(y)):
        for j in range(i+1, len(y)):
            if y[i] == y[j] and t[i] == t[j]:
                a += 1
            elif y[i] == y[j] and t[i] != t[j]:
                b += 1
            elif y[i] != y[j] and t[i] == t[j]:
                c += 1
            elif y[i] != y[j] and t[i] != t[j]:
                d += 1
    return a, b, c, d

def external_index(a, b, c, d, m):
    JC = a / (a + b + c)
    FMI = np.sqrt(a**2 / ((a + b) * (a + c)))
    RI = 2 * ( a + d ) / ( m * (m + 1) )
    return JC, FMI, RI

def evaluate_it(y, t):
    a, b, c, d = evaluate(y, t)
    return external_index(a, b, c, d, len(y))

if __name__ == "__main__":
#数据处理
    df1 = pd.read_excel("data1.xlsx")
    matrix = df1.values
    data = np.array(matrix)
    data_test = np.array(pd.read_excel("datatest.xlsx").values)
#冷却后重量
    X_train = data[200:1900,[8,9,10,11,12,13,14,17]]
    Y_train = data[200:1900,[26]]
    x_bef = data[200:1900,[25]]
    X = data[200:1900,[8,9,10,11,12,13,14,17]]
    Y_true = data[200:1900,[26]]
    x_roller = X_train[:,[0]]
    x_test = data_test[0:1600,[8,9,10,11,12,13,14,17]]
    x_bef_test = data_test[0:1600,[25]]
    y_aft = data_test[0:1600,[26]]
    a = np.array(x_bef)
    b = np.array(Y_train)
    c = a/b
    c = np.append(c,x_roller,axis=1)
    

#PCA降维/PLS
    pls1 = PLSRegression(n_components=3)
    pls1.fit(X_train, c[:,[0]])
    s = pls1.predict(x_test)
    weight = x_bef_test/s #采用收缩率预测后称量质量
    #RMSE1 = mean_squared_error(c[:,[0]],s,squared=False)
    RMSE2 = mean_squared_error(y_aft, weight,squared=False)
    print(RMSE2)
    pca1 = PCA(n_components=2,whiten=True)
    res = pca1.fit_transform(X_train)
    res1 = res[:,[0]]
    res2 = res[:,[1]]
    res1_std = np.array([[-1.7]])
    rbf_res = rbf_kernel(res[:,[0]], np.array([[-1.7]]), gamma=0.1)#核函数提升维度
    ratio = pca1.explained_variance_ratio_
    print("各特征的权重为: ratio = ",ratio)
    print("使用sklearn.decomposition.PCA 验证的结果为: res = ", res)#PCA降维结果输出
    fig1 = plt.figure(1)#绘图展示
    plt.scatter(res[:, 0], res[:, 1],marker='o',c='blue')
    fig = plt.figure(2)#提升维度后变为线性
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(res1[:,0],res2,rbf_res,c='blue')
#FCM聚类
test_y = FCM(c)
#X_reduced = pca1.transform(c)
fig = plt.figure(2)
epochs = list(range(1, len(c) + 1))
plt.scatter(c[:,[1]],c[:,[0]], c=test_y, cmap=plt.cm.Set1)
fig = plt.figure(3)
epochs = list(range(1, len(c) + 1))
plt.plot(epochs,c[:,[0]])
plt.plot(epochs,s,c='red')
from matplotlib.lines import Line2D
custom_lines = [Line2D([0], [0], c="red", lw=4),
                Line2D([0], [0], c="blue", lw=4)]
fig, ax = plt.subplots()
ax.legend(custom_lines, ['预测结果', '实际称量结果'])
plt.xlabel("采样点")
plt.ylabel("称量质量/g")
epochs = list(range(1, len(y_aft) + 1))
epoch = list(range(1, len(weight) + 1))
plt.plot(epoch,weight)
plt.plot(epochs,y_aft)
plt.show()