# src/model.py
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

class ClassifierModel(keras.Model):
    """
    统一的分类器模型 (2-Layer MLP + BN + Dropout)。
    
    兼容性说明:
    1. 结构完全遵循原始脚本: Dense(Relu) -> BN -> Drop -> Dense(Relu) -> BN -> Drop -> Output。
    2. 通过 num_classes 参数区分二分类 (Sigmoid) 和多分类 (Softmax)。
    """
    
    def __init__(self, hidden_dim=128, dropout_rate=0.3, num_classes=1):
        """
        Args:
            hidden_dim: 隐藏层维度 (默认 128)
            dropout_rate: Dropout 比率 (默认 0.3)
            num_classes: 类别数量。
                         - 如果为 1，则输出层为 Sigmoid (用于二分类)。
                         - 如果 > 1，则输出层为 Softmax (用于 Amazon 5分类)。
        """
        super(ClassifierModel, self).__init__()
        
        # Block 1
        self.block1_dense = layers.Dense(hidden_dim, activation='relu')
        self.block1_bn = layers.BatchNormalization()
        self.block1_drop = layers.Dropout(dropout_rate)
        
        # Block 2
        self.block2_dense = layers.Dense(hidden_dim, activation='relu')
        self.block2_bn = layers.BatchNormalization()
        self.block2_drop = layers.Dropout(dropout_rate)
        
        # Output Layer
        if num_classes > 1:
            # Amazon (5-class): Softmax
            self.out = layers.Dense(num_classes, activation='softmax', dtype='float32')
        else:
            # Binary (Reddit, Wiki, Mooc, Epinions): Sigmoid
            self.out = layers.Dense(1, activation='sigmoid', dtype='float32')

    def call(self, x, training=False):
        # Block 1
        x = self.block1_dense(x)
        x = self.block1_bn(x, training=training)
        x = self.block1_drop(x, training=training)
        
        # Block 2
        x = self.block2_dense(x)
        x = self.block2_bn(x, training=training)
        x = self.block2_drop(x, training=training)
        
        return self.out(x)