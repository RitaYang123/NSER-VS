import torch
from torch.nn import functional as F


def soft_threshold(l, x):  #S_lambda(x),其中l是lambda
    return torch.sign(x) * torch.relu(torch.abs(x) - l)


def sign_binary(x):
    ones = torch.ones_like(x) #创建一个与输入张量x相同形状的全1张量
    return torch.where(x >= 0, ones, -ones)  #如果x大于等于0，则元素被替换为1，小于0被替换为-1


def prox(v, u, *, lambda_, lambda_bar, M):   #附录中的推导
    """
    v has shape (m,) or (m, batches)
    u has shape (k,) or (k, batches)

    supports GPU tensors
    """
    onedim = len(v.shape) == 1  #如果v是一维的，则onedim=True
    if onedim:
        v = v.unsqueeze(-1)   #使用unsqueeze将张量的维度从一维变成二维
        u = u.unsqueeze(-1)   #(k,1)或(k,batches)

    u_abs_sorted = torch.sort(u.abs(), dim=0, descending=True).values  #按列排序

    k, batch = u.shape  #batch有可能为1或者c

    s = torch.arange(k + 1.0).view(-1, 1).to(v)#首先生成一个从0到k的序列，然后再进行形状变换为列向量(k+1,1)，.to(v)将这个张量转化成和输入张量v相同的数据类型
    zeros = torch.zeros(1, batch).to(u) #生成一个形状为(1,batch)的零矩阵，同样转化成和u相同的数据类型

    a_s = lambda_ - M * torch.cat(
        [zeros, torch.cumsum(u_abs_sorted - lambda_bar, dim=0)]   #沿着列对张量进行累加求和，然后通过cat将zeros和后面的张量按行拼接
    )  #得到(k+1,batch)或者(k+1,1)

    norm_v = torch.norm(v, p=2, dim=0) #求L2范数

    x = F.relu(1 - a_s / norm_v) / (1 + s * M ** 2)   #x其实是b_s,但是少乘了一个v， relu(x)=max(0,x)

    w = M * x * norm_v  #M|b|_2
    intervals = soft_threshold(lambda_bar, u_abs_sorted)
    lower = torch.cat([intervals, zeros]) #拼接

    idx = torch.sum(lower > w, dim=0).unsqueeze(0) #表示每一列中大于w的元素个数

    #满足约束条件的，x_star好像是求得少乘一个v的b*, w_star是M|b*|_2
    x_star = torch.gather(x, 0, idx).view(1, batch)#选择索引为idx对应未知的元素，并将形状改为(1,batch)
    w_star = torch.gather(w, 0, idx).view(1, batch)

    #theta_star就是W*，其实也就是W
    beta_star = x_star * v  #这里是b*，其实也就是θ
    theta_star = sign_binary(u) * torch.min(soft_threshold(lambda_bar, u.abs()), w_star)

    if onedim:
        beta_star.squeeze_(-1)
        theta_star.squeeze_(-1)

    return beta_star, theta_star
#通过prox我们可以得到优化问题的全局最优解W*和b*

#beta.weight是跳跃(残差)层的权重，也就是prox中的v
def inplace_prox(beta, theta, lambda_, lambda_bar, M):  #theta是权重W，也就是prox中的U
    beta.weight.data, theta.weight.data = prox(
        beta.weight.data, theta.weight.data, lambda_=lambda_, lambda_bar=lambda_bar, M=M
    )


def inplace_group_prox(groups, beta, theta, lambda_, lambda_bar, M):   #这个函数的作用？
    """
    groups is an iterable such that group[i] contains the indices of features in group i
    """
    beta_ = beta.weight.data
    theta_ = theta.weight.data
    beta_ans = torch.empty_like(beta_) #创建与beta_形状相同的张量,beta是跳跃(残差)层权重
    theta_ans = torch.empty_like(theta_)#创建与theta_形状相同的张量，theta是每一层权重
    for g in groups:
        group_beta = beta_[:, g]
        group_beta_shape = group_beta.shape
        group_theta = theta_[:, g]
        group_theta_shape = group_theta.shape
        group_beta, group_theta = prox(
            group_beta.reshape(-1),   #reshape成一个一维向量
            group_theta.reshape(-1),
            lambda_=lambda_,
            lambda_bar=lambda_bar,
            M=M,
        )
        beta_ans[:, g] = group_beta.reshape(*group_beta_shape) #reshape成原来的形状
        theta_ans[:, g] = group_theta.reshape(*group_theta_shape)
    beta.weight.data, theta.weight.data = beta_ans, theta_ans






