import numpy as np
from math import sqrt
from collections import Counter


def kNN_classify(k, X_train, y_train, x):
#k; X, y:训练数据集; x:要预测的特征向量

    #断言 保证传进来的参数合法
    assert 1 <= k <= X_train.shape[0], "k must be valid"
    assert X_train.shape[0] == y_train.shape[0], \
        "the size of X_train must equal to the size of y_train"
    assert X_train.shape[1] == x.shape[0], \
        "the feature number of x must be equal to X_train"

    #kNN算法只有5行
    distances = [sqrt(np.sum((x_train - x)**2)) for x_train in X_train]
    nearest = np.argsort(distances)

    topK_y = [y_train[i] for i in nearest[:k]]
    votes = Counter(topK_y)

    return votes.most_common(1)[0][0]