# -*- coding: utf-8 -*-
"""
Created on Sun May 19 11:06:56 2024

@author: Rui Yang
"""
import numpy as np
import random
from sklearn.metrics.pairwise import euclidean_distances

def generate_er_matrix1(N):
    A=N**-0.5+np.random.random((N,N))#N*N的矩阵
    A=np.floor(A) #向下取整
    A=A-np.diag(np.diag(A))#A减去以矩阵A的对角线元素为对角线的对角矩阵，即对角线元素被设置为0
    return A

def generate_er_matrix2(N):
    A=N**-0.1+np.random.random((N,N))#N*N的矩阵
    A=np.floor(A) #向下取整
    A=A-np.diag(np.diag(A))#A减去以矩阵A的对角线元素为对角线的对角矩阵，即对角线元素被设置为0
    return A

def euclidean_distance(i, j, coordinates):
    """计算节点i和节点j之间的欧氏距离"""
    return np.sqrt((coordinates[i][0] - coordinates[j][0])**2 + (coordinates[i][1] - coordinates[j][1])**2)

def generate_weight_matrix(N, coordinates):
    """基于距离生成N x N空间权重矩阵"""
    
    W = np.zeros((N, N))# 初始化权重矩阵，矩阵的初始值为0
    connection_probability = N ** (-0.1)# 计算连接概率
    
    # 遍历每对节点，决定是否连接
    for i in range(N):
        for j in range(i + 1, N):  # 因为矩阵是对称的，只计算上三角部分
            # 计算节点i和节点j之间的欧氏距离
            dist = euclidean_distance(i, j, coordinates)
            
            # 以概率P(i, j)连接i和j
            if random.random() < connection_probability:
                # 如果连接，设置连接的权重为距离的倒数
                if dist != 0:  # 避免除以零
                    weight = 1 / dist
                    W[i, j] = weight
                    W[j, i] = weight  # 对称性，连接也是双向的
    return W

def simu_multi_norm(x_len, sd=1, pho=0.5):
    V = np.zeros((x_len, x_len))
    for i in range(x_len):
        for j in range(x_len):
            V[i, j] = pho**abs(i - j)
    V *= (sd**2)
    return V


def Generate(N=1000,d=2,rho=0.8,matrix='er0.5',model='linear',err='N_0.64'):
    if matrix=='er0.5':
        A=generate_er_matrix1(N)
        row_sum=np.sum(A,1) #每行元素和
        row_sum[np.argwhere(row_sum==0)]=1 #找到值为0的元素的行的索引，并将其修改为1
        W=(A.T/row_sum).T #行标准化
    elif matrix=='dist':
        coordinates = np.random.rand(N, 2)
        A=generate_weight_matrix(N,coordinates)
        row_sum=np.sum(A,1) #每行元素和
        row_sum[np.argwhere(row_sum==0)]=1 #找到值为0的元素的行的索引，并将其修改为1
        W=(A.T/row_sum).T #行标准化   

    if model=='linear_sparse':
        X0=np.random.normal(0, 1, (N, d))
        X1=np.random.normal(0, 1, (N, d))
        X2=np.random.normal(0, 1, (N, d))
    else:
        X0=np.random.multivariate_normal(mean=np.zeros(d), cov=simu_multi_norm(x_len=d, sd=1, pho=0.5), size=N)
        X1=np.random.multivariate_normal(mean=np.zeros(d), cov=simu_multi_norm(x_len=d, sd=1, pho=0.5), size=N)
        X2=np.random.multivariate_normal(mean=np.zeros(d), cov=simu_multi_norm(x_len=d, sd=1, pho=0.5), size=N)
    
    if err=='N_0.64':
        err0=np.random.normal(0,0.8,(N,1))#随机误差项
        err1=np.random.normal(0,0.8,(N,1))
        err2=np.random.normal(0,0.8,(N,1))
    elif err=='N_2.25':
        err0=np.random.normal(0,1.5,(N,1))#随机误差项
        err1=np.random.normal(0,1.5,(N,1))
        err2=np.random.normal(0,1.5,(N,1))
    elif err=='N_9':
        err0=np.random.normal(0,3,(N,1))#随机误差项
        err1=np.random.normal(0,3,(N,1))
        err2=np.random.normal(0,3,(N,1))
    elif err=='K':
        err0 = np.random.chisquare(1, size=(N, 1))
        err1 = np.random.chisquare(1, size=(N, 1))
        err2 = np.random.chisquare(1, size=(N, 1))
    
    Y0=GenY(W,X0,rho,err0,model)
    Y1=GenY(W,X1,rho,err1,model)
    Y2=GenY(W,X2,rho,err2,model)

    
    return [W,A,X0,Y0,err0],[W,A,X1,Y1,err1],[W,A,X2,Y2,err2]

def GenY(W,X,rho,err,model='linear'):
    N=len(err)
    if model=='linear_sparse':
        d=X.shape[1]
        beta = np.array([1.8, 2, 0, 0, 1.5, 0, 0, 0, 0, 0]).reshape(-1, 1)
        I=np.eye(N)
        Y=np.dot(np.linalg.inv(I-rho*W),np.dot(X,beta)+err)

    elif model=='nonparametric':
        GX=1.5*np.log(abs(X[:,0]))+2*np.sin(X[:,2]+X[:,4])
        GX=GX[np.newaxis].T#增加一个维度，(n,1)
        
        I=np.eye(N)
        Y=np.dot(np.linalg.inv(I-rho*W),GX+err)   
        
   
    return Y
    
    
    

