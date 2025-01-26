from model import LassoNet
from itertools import islice
from abc import ABCMeta, abstractmethod, abstractstaticmethod
from dataclasses import dataclass
from functools import partial
import itertools
from typing import List
import warnings
import matplotlib.pyplot as plt
import pandas as pd
import math
import numpy as np
from sklearn.base import (
    BaseEstimator,
    ClassifierMixin,
    MultiOutputMixin,
    RegressorMixin,
)
from sklearn.model_selection import check_cv, train_test_split
import torch
from torch import nn
import torch.nn.functional as F
from torch.nn.parameter import Parameter
from tqdm import tqdm
#import GenWXY_sparse
#from lassonet.interfaces import BaseLassoNet


#from .cox import CoxPHLoss, concordance_index
#, metaclass=ABCMeta

class SAR_ExpectileLoss(nn.Module):
    def __init__(self, expectile):
        super(SAR_ExpectileLoss, self).__init__()
        self.expectile = expectile

    def forward(self, X, Y, f, W, rho):
        n = Y.shape[0]
        I = torch.eye(n)  # 生成单位矩阵
        #G = I - rho * torch.tensor(W, requires_grad=True)
        G = I - rho * (W.clone().detach()).requires_grad_(True)
        G = G.double()
        Y = Y.double()
        S=torch.mm(G, Y) - f(X)
        S_2 = torch.pow(S, 2)
        expectile_loss = torch.where(S >= 0, self.expectile * S_2, (1 - self.expectile) * S_2)
        return expectile_loss.mean()


@dataclass
class HistoryItem:
    lambda_: float
    rho: float
    state_dict: dict
    objective: float  # loss + lambda_ * regularization
    loss: float
    val_objective: float  # val_loss + lambda_ * regularization
    val_loss: float
    regularization: float
    l2_regularization: float
    l2_regularization_skip: float
    selected: torch.BoolTensor
    n_iters: int

    def log(item):
        print(
            f"{item.n_iters} epochs, "
            f"val_objective "
            f"{item.val_objective:.2e}, "
            f"val_loss "
            f"{item.val_loss:.2e}, "
            f"regularization {item.regularization:.2e}, "
            f"l2_regularization {item.l2_regularization:.2e}"
        )


class E_SAR_LassoNet(BaseEstimator, metaclass=ABCMeta):#BaseEstimator是一个基类，ABCMeta是Python中用于创建抽象基类的元类
    def __init__(
            self,
            *,
            hidden_dims=(100,),
            lambda_start="auto",
            lambda_seq=None,
            gamma=0.0,
            gamma_skip=0.0,
            path_multiplier=1.02,
            M=10,
            dropout=0,
            batch_size=None,
            optim=None,
            n_iters=(1000, 100),  # (1000,100)
            patience=(100, 10),
            tol=0.99,
            backtrack=False,
            val_size=None,
            device=None,
            verbose=1,
            random_state=None,
            torch_seed=None,
            #tie_approximation=None,
            expectile=0.5,
    ):
        """
        Parameters
        ----------
        hidden_dims : tuple of int, default=(100,)
            Shape of the hidden layers.
        lambda_start : float, default='auto'
            First value on the path. Leave 'auto' to estimate it automatically.
        lambda_seq : iterable of float
            If specified, the model will be trained on this sequence
            of values, until all coefficients are zero.
            The dense model will always be trained first.
            Note: lambda_start and path_multiplier will be ignored.
        gamma : float, default=0.0
            l2 penalization on the network
        gamma_skip : float, default=0.0
            l2 penalization on the skip connection
        path_multiplier : float, default=1.02
            Multiplicative factor (:math:`1 + \\epsilon`) to increase
            the penalty parameter over the path
        M : float, default=10.0
            Hierarchy parameter.
        dropout : float, default = None
        batch_size : int, default=None
            If None, does not use batches. Batches are shuffled at each epoch.
        optim : torch optimizer or tuple of 2 optimizers, default=None
            Optimizer for initial training and path computation.
            Default is Adam(lr=1e-3), SGD(lr=1e-3, momentum=0.9).
        n_iters : int or pair of int, default=(1000, 100)
            Maximum number of training epochs for initial training and path computation.
            This is an upper-bound on the effective number of epochs, since the model
            uses early stopping.
        patience : int or pair of int or None, default=10
            Number of epochs to wait without improvement during early stopping.
        tol : float, default=0.99
            Minimum improvement for early stopping: new objective < tol * old objective.
        backtrack : bool, default=False
            If true, ensures the objective function decreases.
        val_size : float, default=None
            Proportion of data to use for early stopping.
            0 means that training data is used.
            To disable early stopping, set patience=None.
            Default is 0.1 for all models except Cox for which training data is used.
            If X_val and y_val are given during training, it will be ignored.
        device : torch device, default=None
            Device on which to train the model using PyTorch.
            Default: GPU if available else CPU
        verbose : int, default=1
        random_state
            Random state for validation
        torch_seed
            Torch state for model random initialization
        class_weight : iterable of float, default=None
            If specified, weights for different classes in training.
            There must be one number per class.
        tie_approximation: str
            Tie approximation for the Cox model, must be one of ("breslow", "efron").
        expectile: float
            expectile value, default=0.5.
        """
        assert isinstance(hidden_dims, tuple), "`hidden_dims` must be a tuple"
        self.hidden_dims = hidden_dims
        self.lambda_start = lambda_start
        self.lambda_seq = lambda_seq
        self.gamma = gamma
        self.gamma_skip = gamma_skip
        self.path_multiplier = path_multiplier
        self.M = M
        self.dropout = dropout
        self.batch_size = batch_size
        self.optim = optim
        if optim is None:   #可以选择Adam或者SGD
            optim = (
                partial(torch.optim.Adam, lr=0.01),
                partial(torch.optim.SGD, lr=0.01, momentum=0.9),
            )
        if isinstance(optim, partial): #判断optim是不是partial函数
            optim = (optim, optim)
        self.optim_init, self.optim_path = optim  #optim_init用于初始化模型参数时使用的优化器对象，optim_path用于训练过程中更新模型参数时使用的优化器对象
        if isinstance(n_iters, int):
            n_iters = (n_iters, n_iters)
        self.n_iters = self.n_iters_init, self.n_iters_path = n_iters #n_iters_init 则是类的成员变量，用于存储模型参数初始化时的迭代次数。n_iters 是在类的构造函数中传递进来的参数，它是一个元组，包含了模型参数初始化和训练过程中的迭代次数。
        if patience is None or isinstance(patience, int):
            patience = (patience, patience)
        self.patience = self.patience_init, self.patience_path = patience
        self.tol = tol
        self.backtrack = backtrack
        if val_size is None:
            # TODO: use a cv parameter following sklearn's interface
            val_size = 0.1
        self.val_size = val_size

        if device is None:
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.device = device

        self.verbose = verbose

        self.random_state = random_state
        self.torch_seed = torch_seed
        self.expectile = expectile

        self.model = None

        self.criterion = SAR_ExpectileLoss(expectile)

        @abstractmethod
        def _convert_y(self, y) -> torch.TensorType:
            """Convert y to torch tensor"""
            raise NotImplementedError
    #
    #    @abstractstaticmethod
    #    def _output_shape(cls, y):
    #        """Number of model outputs"""
    #        raise NotImplementedError
    #
    #    @abstractattr

    def _init_model(self, X, W):
        """Create a torch model"""
        if self.torch_seed is not None:
            torch.manual_seed(self.torch_seed) #设置为pytorch随机种子
        self.model = LassoNet(
            X.shape[1], *self.hidden_dims, 1, dropout=self.dropout
        ).to(self.device)
        self.W = W #空间权重矩阵
        self.rho = Parameter(torch.Tensor([0.5])) #ρ=0.5
        self.param=nn.ParameterList([self.rho])
        # self.param = nn.ParameterList([self.rho, self.sigma2])
        self.param.extend(self.model.parameters()) #model.parameter会自动获取模型中所有需要学习的参数，这里加入了ρ

    def _cast_input(self, X, y=None):
        if hasattr(X, "to_numpy"): #检查x是否具有to_numpy方法
            X = X.to_numpy() #将X转换成numpy数组
        X = torch.FloatTensor(X).to(self.device) #将X转换为浮点类型
        if y is None:
            return X
        if hasattr(y, "to_numpy"):
            y = y.to_numpy()
        y = self._convert_y(y)    #将y转换成适合模式处理的格式
        return X, y

    def fit(self, X, y, W, *, X_val=None, y_val=None):#x_val是可选参数
        """Train the model.
        Note that if `lambda_` is not given, the trained model
        will most likely not use any feature.
        """
        self.path_ = self.path(X, y, W, X_val=X_val, y_val=y_val, return_state_dicts=False)
        return self

    def _train(
            self,
            X_train,
            y_train,
            W,
            X_val,
            y_val,
            *,
            batch_size,
            epochs,
            lambda_,
            optimizer,
            return_state_dict,
            patience=None,
    ) -> HistoryItem:  #historyitem会保留每个epoch的训练、验证损失，训练和验证指标等信息

        model = self.model

        def validation_obj():
            with torch.no_grad():
                return (
                        self.criterion(X_val, y_val, model, W, self.rho).item()
                        + lambda_ * model.l1_regularization_skip().item()
                        + self.gamma * model.l2_regularization().item()
                        + self.gamma_skip * model.l2_regularization_skip().item()
                )

        best_val_obj = validation_obj()
        epochs_since_best_val_obj = 0
        if self.backtrack:  #如果是TRUE，代表目标函数值减小，初始设的False
            best_state_dict = self.model.cpu_state_dict()
            real_best_val_obj = best_val_obj
            real_loss = float("nan")  # if epochs == 0

        n_iters = 0

        n_train = len(X_train)
        if batch_size is None:
            batch_size = n_train
            randperm = torch.arange
        else:
            randperm = torch.randperm  #生成一个随机排序的索引向量，用于训练时将训练数据打乱
        batch_size = min(batch_size, n_train)

        for epoch in range(epochs):
            indices = randperm(n_train)
            model.train()
            loss = 0
            for i in range(n_train // batch_size):   #分成批次进行训练
                # don't take batches that are not full
                batch = indices[i * batch_size: (i + 1) * batch_size]

                def closure():
                    nonlocal loss   #非局部变量
                    optimizer.zero_grad()  #清除之前的梯度信息
                    ans = (
                        # self.criterion(model(X_train[batch]), y_train[batch])
                        self.criterion(X_train, y_train, model, W, self.rho)
                        + self.gamma * model.l2_regularization()
                        + self.gamma_skip * model.l2_regularization_skip()
                    )
                    ans.backward() #反向传播,pytorch自动求导不需要手动指定学习率
                    loss += ans.item() * len(batch) / n_train #不同批次为了得到一个相同量级的损失值
                    return ans

                #反向传播更新参数
                optimizer.step(closure)  #表示使用优化器来进行参数更新，optimizer.param_groups[0]代表选择的是Adam
                model.prox(lambda_=lambda_ * optimizer.param_groups[0]["lr"], M=self.M)  #这个地方HIER-PROX中lambda乘了学习率α

            if epoch == 0:
                # s
                real_loss = loss
            val_obj = validation_obj()
            if val_obj < self.tol * best_val_obj:
                best_val_obj = val_obj
                epochs_since_best_val_obj = 0
            else:
                epochs_since_best_val_obj += 1
            if self.backtrack and val_obj < real_best_val_obj:
                best_state_dict = self.model.cpu_state_dict()
                real_best_val_obj = val_obj
                real_loss = loss
                n_iters = epoch + 1
            if patience is not None and epochs_since_best_val_obj == patience:
                break

        if self.backtrack:#如果是TRUE，代表目标函数值减小，初始设的False
            self.model.load_state_dict(best_state_dict)
            val_obj = real_best_val_obj
            loss = real_loss
        else:
            n_iters = epoch + 1
        with torch.no_grad():
            reg = self.model.l1_regularization_skip().item() #对跳跃(残差)层的权重求L2范数和
            l2_regularization = self.model.l2_regularization()  #除第一层外的权重
            l2_regularization_skip = self.model.l2_regularization_skip()
        return HistoryItem(
            lambda_=lambda_,
            rho=torch.tanh(self.rho).item(),
            state_dict=self.model.cpu_state_dict() if return_state_dict else None,
            objective=loss + lambda_ * reg,   #目标函数
            loss=loss,
            val_objective=val_obj, #目标函数
            val_loss=val_obj - lambda_ * reg, #损失函数
            regularization=reg,    #跳跃层的权重L2范数和
            l2_regularization=l2_regularization,   #除第一层外的权重的L2范数和
            l2_regularization_skip=l2_regularization_skip, #跳跃层的权重L2范数
            selected=self.model.input_mask().cpu(),   #看特征有没有被选择并把结果转移到CPU上
            n_iters=n_iters,
        )

    #    @abstractmethod
    #    def predict(self, X):
    #        raise NotImplementedError

    @staticmethod
    def _compute_feature_importances(path: List[HistoryItem]):#path是一个列表，其中的元素是historyitem的对象，确保使用path变量时，只能存储和访问historyitem类型的对象
        """When does each feature disappear on the path?

        Parameters
        ----------
        path : List[HistoryItem]

        Returns
        -------
            feature_importances_
        """

        current = path[0].selected.clone() #访问第一个历史记录，获取path列表中第一个元素的selected属性，并进行克隆，包含了所有特征
        ans = torch.full(current.shape, float("inf")) #生成一个与current.shape一样的张量，所有元素都被设置为正无穷大
        for save in islice(path, 1, None): #从索引1开始到末尾的列表元素，即忽略第一个元素
            lambda_ = save.lambda_   #取出当前迭代的lamda_值
            diff = current & ~save.selected #计算出 current和save.selected按位取反的结果，并赋值给diff。这样可以获取在current中存在而在save.selected中不存在的位置
            ans[diff.nonzero().flatten()] = lambda_  #diff.nonzero().flatten()找到非零元素的索引，并将这些位置赋值为lambda的值，计算特征消失的时间
            current &= save.selected #按位与操作，将current更新为仅包含在current和save.selected中都存在的特征
        return ans  #记录了每个特征消失的时间

    def path(
            self,
            X,
            y,
            W,
            *,
            X_val=None,
            y_val=None,
            lambda_seq=None,
            lambda_max=float("inf"), #初值无穷大
            return_state_dicts=True,
            callback=None#,f
    ) -> List[HistoryItem]:
        """Train LassoNet on a lambda_ path.
        The path is defined by the class parameters:
        start at `lambda_start` and increment according to `path_multiplier`.
        The path will stop when no feature is being used anymore.
        callback will be called at each step on (model, history)
        """
        #        assert (X_val is None) == (
        #            y_val is None
        #        ), "You must specify both or none of X_val and y_val"
        #        sample_val = self.val_size != 0 and X_val is None
        #        if sample_val:
        #            X_train, X_val, y_train, y_val = train_test_split(
        #                X, y, test_size=self.val_size, random_state=self.random_state
        #            )
        #        elif X_val is None:
        #            X_train, y_train = X_val, y_val = X, y
        #        else:
        #            X_train, y_train = X, y
        X_train, y_train = X_val, y_val = X, y
        # X_train, y_train = self._cast_input(X_train, y_train)
        # X_val, y_val = self._cast_input(X_val, y_val)

        hist: List[HistoryItem] = [] #空列表，其中的元素类型为HistoryItem

        # always init model
        self._init_model(X_train, W)

        hist.append(
            self._train(
                X_train,
                y_train,
                W,
                X_val,
                y_val,
                batch_size=self.batch_size,
                lambda_=0,
                epochs=self.n_iters_init,
                optimizer=self.optim_init(t for t in self.param),  ###
                patience=self.patience_init,
                return_state_dict=return_state_dicts,
            )
        )
        if callback is not None:
            callback(self, hist)   #callback函数实时打印损失值，记录训练指标、保存模型
        if self.verbose > 1:
            print("Initialized dense model")
            hist[-1].log()  #hist[-1]时迭代次数，historyitem里面有个log函数

        optimizer = self.optim_path(t for t in self.param)

        # build lambda_seq
        if lambda_seq is not None:
            pass
        elif self.lambda_seq is not None:
            lambda_seq = self.lambda_seq
        else:

            def _lambda_seq(start):
                while start <= lambda_max:  #lambda_max是无穷大
                    yield start  #每次迭代时产生一个值并将这个值返回给调用者，所以返回的是序列
                    start *= self.path_multiplier  #其实这个地方可以看作设置的epsilon为0.02

            if self.lambda_start == "auto":
                # divide by 10 for initial training
                self.lambda_start_ = (
                        self.model.lambda_start(M=self.M)
                        / optimizer.param_groups[0]["lr"]   #Adam的学习率
                        / 10   #除以这两个数是为了得到一个适当的正则化参数的初始值？
                )#除以学习率可以使得正则化参数的初始值与学习率相适应，从而更好的控制模型复杂度？除以10是一种经验性的做法，可以使得正则化参数的初始值不过大，避免在训练初期对模型造成过多的约束
                lambda_seq = _lambda_seq(self.lambda_start_)
            else:
                lambda_seq = _lambda_seq(self.lambda_start)

        # extract first value of lambda_seq
        lambda_seq = iter(lambda_seq) #将lambda_seq中生成的lambda值序列转换成一个迭代器
        lambda_start = next(lambda_seq)#从迭代器中获取下一个值

        is_dense = True
        for current_lambda in itertools.chain([lambda_start], lambda_seq):  #current_lambda从迭代器中依次取出值
            if self.model.selected_count() == 0:
                break
            last = self._train(  #返回损失值什么的
                X_train,
                y_train,
                W,
                X_val,
                y_val,
                batch_size=self.batch_size,
                lambda_=current_lambda,
                epochs=self.n_iters_path,
                optimizer=optimizer,
                patience=self.patience_path,
                return_state_dict=return_state_dicts,
            )
            if is_dense and self.model.selected_count() < X.shape[1]: #小于x的维数
                is_dense = False
                #if current_lambda / lambda_start < 2: #为什么是2
                #    warnings.warn(
                #        f"lambda_start={self.lambda_start:.3f} "
                #        "might be too large.\n"
                #        f"Features start to disappear at {current_lambda:.3f}."
                #    )

            hist.append(last)
            if callback is not None:#callback函数实时打印损失值，记录训练指标、保存模型
                callback(self, hist)

            if self.verbose > 1:#如果条件成立，就会输出一条包含当前正则化参数值和模型选择的特征数量的信息。
                print(
                    f"Lambda = {current_lambda:.2e}, "
                    f"selected {self.model.selected_count()} features "
                )
                last.log()
        ###
        self.feature_importances_ = self._compute_feature_importances(hist)
        """When does each feature disappear on the path?"""

        return hist

    def load(self, state_dict):
        if isinstance(state_dict, HistoryItem):
            state_dict = state_dict.state_dict
        if self.model is None:
            output_shape, input_shape = state_dict["skip.weight"].shape
            self.model = LassoNet(
                input_shape,
                *self.hidden_dims,
                output_shape,
                # groups=self.groups,
                dropout=self.dropout,
            ).to(self.device)

        self.model.load_state_dict(state_dict)
        return self


class LassoNetSARExpectileRegressor(
    RegressorMixin,
    MultiOutputMixin,
    E_SAR_LassoNet,
):
    """Use LassoNet as regressor"""

    #    def _convert_y(self, y):
    #        y = torch.FloatTensor(y).to(self.device)
    #        if len(y.shape) == 1:
    #            y = y.view(-1, 1)
    #        return y
    #
    #    @staticmethod
    #    def _output_shape(y):
    #        return y.shape[1]
    #
    # criterion = MLE_loss()

    def predict(self, X):
        self.model.eval()#将模型设置为评估模式，模型会禁用一些具有随机性质的操作
        with torch.no_grad():
            ans = self.model(self._cast_input(X))#self._cast_input(X)将输入X转换为模型所期望的数据类型
        if isinstance(X, np.ndarray):#检查X的数据类型是否为np.ndarray类型
            ans = ans.cpu().numpy()
        return ans

    def score(self, X_test, y_test, rho, W):
        """Mean Squared Error"""
        # Make predictions
        predictions = self.predict(X_test)
        N=X_test.shape[0]
        I_tensor = torch.tensor(np.eye(N), dtype=torch.float32)  # Convert to PyTorch tensor

        prediction = torch.matmul(torch.linalg.inv(I_tensor - rho * W), predictions)
        # Extract true labels
        true_labels = y_test  # Assuming y_test contains the true labels directly
        # Compute Mean Squared Error
        mse = torch.mean((prediction - true_labels) ** 2)
        return mse




