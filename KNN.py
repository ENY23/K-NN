"""
K-近邻 (K-Nearest Neighbors, K-NN) 分类器实现模块
该模块实现了经典的KNN分类算法，使用纯Python和NumPy库。
包含KNN类的完整实现和一个演示用的主函数。
"""
#K-近邻 (K-Nearest Neighbors, K-NN) 分类器实现，
import time
import numpy as np
from scipy import stats
import pandas as pd

class KNN:
    """
    K-近邻分类器
    通过计算测试样本与训练样本之间的欧几里得距离，
    找到最近的k个邻居，根据多数投票原则预测类别。
    Attributes:
        n_neighbors (int): 最近邻的数量
        _X_train (np.ndarray): 训练特征数据
        _y_train (np.ndarray): 训练标签数据
    """

    #初始化KNN分类器
    def __init__(self, n_neighbors: int):
        """
        初始化KNN分类器
        Args:
            n_neighbors: 最近邻的数量，必须为正整数
        Raises:
            ValueError: 当n_neighbors不是正整数时抛出异常
        """
        if n_neighbors <= 0:
            raise ValueError("n_neighbors必须是正整数")
        self.n_neighbors = n_neighbors
        self._X_train = None
        self._y_train = None
        
    #训练KNN模型（存储训练数据）
    def fit(self, X_train, y_train):
        """
        训练KNN模型（存储训练数据）
        KNN是一种惰性学习算法，训练阶段只是存储数据，
        真正的计算发生在预测阶段。
        Args:
            X_train: 训练特征数据，形状为(n_samples, n_features)
            y_train: 训练标签数据，形状为(n_samples,)
        Returns:
            self: 返回模型实例本身，支持链式调用
        """
        self._X_train = X_train
        self._y_train = y_train
        
        return self
    
    def predict(self, X_test):
        """
        预测测试数据的类别
        对于每个测试样本：
        1. 计算与所有训练样本的欧几里得距离
        2. 选择距离最近的k个邻居
        3. 通过多数投票确定预测类别
        Args:
            X_test: 测试特征数据，形状为(n_test_samples, n_features)
        Returns:
            np.ndarray: 预测的类别标签，形状为(n_test_samples,)
        Raises:
            ValueError: 当模型未训练或测试数据为空时抛出异常
        """
        #输入验证
        if self._X_train is None or self._y_train is None:
            raise ValueError("模型必须先训练再进行预测")
        
        if len(X_test) == 0:
            raise ValueError("测试数据不能为空")
        
        if self.n_neighbors > len(self._X_train):
            raise ValueError("最近邻数量不能大于训练样本数量")
        
        predictions = []
        
        # 遍历每个测试样本
        for x_test in X_test.values:
            #计算当前测试样本与所有训练样本的欧几里得距离
            #ord=2表示使用L2范数（欧几里得距离），axis=1表示按行计算
            distances = np.linalg.norm(x_test - self._X_train, ord=2, axis=1)
            
            #对距离进行排序，获取最近的前k个邻居的索引
            nearest_indices = np.argsort(distances)[:self.n_neighbors]
            
            #获取这些最近邻居的标签
            neighbor_labels = self._y_train[nearest_indices]
            
            #使用多数投票确定预测类别
            #mode函数返回出现次数最多的标签
            most_common_label = stats.mode(neighbor_labels, axis=None)[0]
            
            predictions.append(most_common_label)
        
        return np.array(predictions)

def main():
    """
    主函数：演示KNN分类器的使用
    
    创建一个简单的二分类问题示例，展示KNN的完整工作流程。
    包括数据准备、模型训练、预测和结果验证。
    """
    #设置numpy显示选项，避免科学计数法显示
    np.set_printoptions(suppress=True)
    
    print("=== K-近邻分类器演示 ===")
    
    #创建训练数据：简单的二维特征二分类问题
    #前3个样本属于类别0，后3个样本属于类别1
    X_train = pd.DataFrame([
        [1.2, 3.1],  
        [2.3, 4.2],   
        [1.1, 2.9],  
        [8.7, 7.6],  
        [9.2, 8.1],  
        [7.9, 8.8]   
    ])
    
    #对应的训练标签
    y_train = np.array([0, 0, 0, 1, 1, 1])
    
    #创建测试数据：两个测试样本
    X_test = pd.DataFrame([
        [1.5, 3.2],  #测试样本1，预期属于类别0（靠近前3个训练样本）
        [8.5, 8.0]   #测试样本2，预期属于类别1（靠近后3个训练样本）
    ])
    
    print(f"训练数据形状: {X_train.shape}")
    print(f"训练标签: {y_train}")
    print(f"测试数据形状: {X_test.shape}")
    
    #初始化KNN模型，设置k=3
    print(f"\n初始化KNN模型 (k={3})...")
    knn = KNN(n_neighbors=3)
    
    #训练模型
    print("训练模型中...")
    knn.fit(X_train, y_train)
    
    #预测测试数据
    print("进行预测...")
    start_time = time.time()
    predictions = knn.predict(X_test)
    end_time = time.time()
    
    #输出结果
    print(f"\n=== 预测结果 ===")
    print(f"测试数据的预测结果: {predictions}")
    print(f"算法运行时间: {end_time - start_time:.6f}秒")
    
    #验证预测结果是否符合预期
    expected = np.array([0, 1])
    if np.array_equal(predictions, expected):
        print("✅ 预测结果符合预期！")
    else:
        print("❌ 预测结果与预期不符！")


if __name__ == "__main__":
    #当直接运行此脚本时执行主函数
    main()