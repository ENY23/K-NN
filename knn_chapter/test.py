"""
KNN分类器在鸢尾花数据集上的验证测试

使用真实的鸢尾花数据集验证手搓KNN算法的性能，
包含完整的机器学习工作流程和评估指标。
"""

import time
import numpy as np
import pandas as pd
from sklearn.datasets import load_iris  # 加载示例数据集
from sklearn.model_selection import train_test_split  # 拆分训练集和测试集
from sklearn.metrics import (  # 评估指标
    accuracy_score,
    confusion_matrix,
    classification_report
)


class KNN(object):
    """
    K-近邻分类器
    
    使用纯NumPy实现的KNN分类算法，支持多分类问题。
    基于欧氏距离和多数投票机制进行分类预测。
    """
    
    def __init__(self, n_neighbors):
        """
        初始化KNN分类器
        
        Args:
            n_neighbors (int): 最近邻的数量
        """
        self.n_neighbors = n_neighbors
        self._X_train = None 
        self._y_train = None
        
    def fit(self, X_train, y_train):
        """
        训练KNN模型
        
        Args:
            X_train: 训练特征数据
            y_train: 训练标签数据
            
        Returns:
            self: 返回模型实例本身
        """
        self._X_train = X_train
        self._y_train = y_train
        return self 
            
    def predict(self, X_test):
        """
        预测测试数据的类别
        
        Args:
            X_test: 测试特征数据
            
        Returns:
            np.ndarray: 预测的类别标签
        """
        # 计算测试样本与所有训练样本的欧氏距离
        distances = [np.linalg.norm(x_test - self._X_train, ord=2, axis=1) for x_test in X_test.values]
        # 按距离排序，取前n_neighbors个邻居的索引
        distances_sort = [np.argsort(distance)[0:self.n_neighbors] for distance in distances]
        # 获取邻居的标签
        target = [self._y_train[distance_s] for distance_s in distances_sort]
        # 投票决定最终类别（取众数）
        class_result = np.array([np.argmax(np.bincount(t)) for t in target])  # 替换scipy的mode，避免依赖
        return class_result


def main():
    """
    主函数：执行完整的KNN模型验证流程
    
    包括数据加载、模型训练、预测和评估等步骤。
    """
    # 1. 加载数据集（以鸢尾花数据集为例，包含3类花，特征为4维）
    iris = load_iris()
    X = pd.DataFrame(iris.data, columns=iris.feature_names)  # 特征（150个样本，4个特征）
    y = iris.target  # 标签（0,1,2代表3种花）

    # 2. 拆分训练集（80%）和测试集（20%），random_state固定随机种子，保证结果可复现
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    # 3. 初始化并训练KNN模型（k=3）
    knn = KNN(n_neighbors=3)
    knn.fit(X_train, y_train)

    # 4. 预测测试集
    start_time = time.time()
    y_pred = knn.predict(X_test)  # 预测结果
    end_time = time.time()

    # 5. 计算评估指标
    print(f"算法运行时间：{end_time - start_time:.6f}秒")
    print("\n1. 准确率（Accuracy）：", accuracy_score(y_test, y_pred))  # 正确预测的比例

    print("\n2. 混淆矩阵：")
    print(confusion_matrix(y_test, y_pred))  # 行：真实标签，列：预测标签

    print("\n3. 详细分类报告：")
    print(classification_report(y_test, y_pred))  # 包含精确率、召回率、F1分数等


if __name__ == "__main__":
    main()