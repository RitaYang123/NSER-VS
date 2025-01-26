from itertools import islice

import torch
from torch import nn
from torch.nn import functional as F
import prox
from prox import inplace_prox, prox


class LassoNet(nn.Module):
    def __init__(self, *dims, dropout=None): #*dims是可变数量的参数，可以接收任意数量的参数
        """
        first dimension is input
        last dimension is output
        """
        assert len(dims) > 2  #判断dims的长度是否大于2,如果不满足则抛出异常
        super().__init__()
        #通过传入的参数dropout来判断是否需要添加dropout层
        self.dropout = nn.Dropout(p=dropout) if dropout is not None else None
        self.layers = nn.ModuleList(   #这里的维度就可以看作神经元的个数
            [nn.Linear(dims[i], dims[i + 1]) for i in range(len(dims) - 1)]
        )
        #创建多个线性层的nn.ModuleList,线性层的输入维度为dims[i],输出维度为dims[i+1],用于构建神经网络模型的隐藏层
        #pytorch中nn.Linear会自动创建每个神经元的偏置向量，所以包含权重参数和偏置向量两个参数
        #没有偏置项的跳跃（残差）层,将输入的第一个维度和最后一个维度进行线性变换，用于连接输入和输出
        self.skip = nn.Linear(dims[0], dims[-1], bias=False)

    def forward(self, inp):
        current_layer = inp
        result = self.skip(inp) #利用跳跃（残差）层进行线性变换得到result,实现了跳跃连接，将输入直接连接到输出，得到θx
        for theta in self.layers:  #每个线性层theta对current_layer进行线性变换，theta是每一层的参数,其实也就是Wx+b
            current_layer = theta(current_layer)
            if theta is not self.layers[-1]:   #如果不是最后一层，则进行Relu激活
                if self.dropout is not None:
                    current_layer = self.dropout(current_layer)
                current_layer = F.relu(current_layer)
        return result + current_layer    #跳跃连接的结果和经过隐藏层处理后的结果相加作为最终输出即θx+f(x)

    def prox(self, *, lambda_, lambda_bar=0, M=1):
        with torch.no_grad(): #上下文管理器，确保在执行prox方法时不进行梯度计算
            inplace_prox(     #inplace_prox函数实现了对网络参数进行L1正则化的近端算子操作
                beta=self.skip,
                theta=self.layers[0],   #第一个隐藏层权重
                lambda_=lambda_,         #正则化参数的标量
                lambda_bar=lambda_bar,   #对偶变量的标量
                M=M,
            )
#使用prox方法可以对lassonet模型的网络参数进行L1正则化近端算子操作，以促使模型学习稀疏的特征表示

    def lambda_start(   #用于估计模型开始稀疏化的阈值
        self,
        M=1,
        lambda_bar=0,
        factor=2,
    ):
        """Estimate when the model will start to sparsify."""

        def is_sparse(lambda_):  #用于判断模型参数是否已经变得稀疏，通过循环迭代调用prox函数来更新模型参数，直到模型参数不再发生变化或达到一定的迭代次数
            with torch.no_grad():
                beta = self.skip.weight.data  #θ
                theta = self.layers[0].weight.data  #第一个隐藏层的权重

                for _ in range(10000):
                    new_beta, theta = prox(
                        beta,
                        theta,
                        lambda_=lambda_,
                        lambda_bar=lambda_bar,
                        M=M,
                    )
                    if torch.abs(beta - new_beta).max() < 1e-5:  #beta是跳跃(残差)层的参数，通过θ是否为0判断是否选择了特征,如果最大值都是0了，说明再次循环选择的特征没有变化
                        # print(_)
                        break
                    beta = new_beta  #新一次循环更新的θ赋值
                return (torch.norm(beta, p=2, dim=0) == 0).sum() #按列计算参数beta的L2范数，并统计L2范数为0的元素的个数，从而判断参数的稀疏性，直到

        start = 1e-6
        while not is_sparse(factor * start):  #如果is_sparse返回false，即模型不是稀疏的，则继续循环，这个乘积代表λ
            start *= factor   #循环逐步增加start的值
        return start

    def l2_regularization(self):   #遍历除了第一层以外的所有层，计算了每一层权重的L2范数的平方
        """
        L2 regulatization of the MLP without the first layer
        which is bounded by the skip connection
        """
        ans = 0
        for layer in islice(self.layers, 1, None):  #从获取索引为1到最后一个索引的所有元素
            ans += (     #将每一层权重的L2范数的平方累加到变量ans中
                torch.norm(
                    layer.weight.data,    #神经网络的权重参数
                    p=2,
                )
                ** 2
            )
        return ans

    def l1_regularization_skip(self): #跳跃（残差）层权重参数每一列的L2范数之和
        return torch.norm(self.skip.weight.data, p=2, dim=0).sum()

    def l2_regularization_skip(self):#跳跃（残差）层权重参数的L2正则化
        return torch.norm(self.skip.weight.data, p=2)

    def input_mask(self):
        with torch.no_grad():#计算每一列的L2范数，如果不为0，则返回True
            return torch.norm(self.skip.weight.data, p=2, dim=0) != 0
#这里主要是看θ是不是0，如果是0则这个特征没有被选择，不是0的话说明被选择了

    def selected_count(self): #计算选择的特征数量
        return self.input_mask().sum().item()

    def cpu_state_dict(self): #将模型的参数转移到CPU并返回参数的状态字典
        return {k: v.detach().clone().cpu() for k, v in self.state_dict().items()}







