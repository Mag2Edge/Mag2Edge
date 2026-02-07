# main.py
import os
import gc
import argparse
import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from tensorflow import keras

# 引入自定义模块
from src.utils import (
    seed_everything, 
    config_gpu, 
    evaluate_binary_predictions, 
    evaluate_multiclass_predictions
)
from src.dataloader import load_data
from src.solver import MagneticHadamardSolver
from src.model import ClassifierModel

# -----------------------------------------------------------------------------
# 全局配置 (Global Configuration)
# -----------------------------------------------------------------------------
SEEDS = [42, 43, 44, 45, 46]
EMBEDDING_DIM = 128
MAGNETIC_Q = 0.25

# 训练超参数
TOTAL_EPOCHS = 150
LEARNING_RATE = 0.001
PATIENCE = 20
BATCH_SIZE = 2048
HIDDEN_DIM = 128
DROPOUT_RATE = 0.3

# 数据集特定配置 (Tolerance, Max Iterations, Number of Classes)
DATASET_CONFIGS = {
    'reddit':       {'tol': 1e-8, 'max_iter': 5000, 'num_classes': 1}, # Binary
    'wikiconflict': {'tol': 1e-6, 'max_iter': 5000, 'num_classes': 1}, # Binary
    'amazon':       {'tol': 1e-5, 'max_iter': 3000, 'num_classes': 5}, # Multiclass (0-4)
    'mooc':         {'tol': 1e-6, 'max_iter': 5000, 'num_classes': 1}, # Binary
    'epinions':     {'tol': 1e-5, 'max_iter': 5000, 'num_classes': 1}  # Binary
}

def run_experiment(dataset_name, data_dir, gpu_id):
    # 1. 配置环境
    config_gpu(gpu_id)
    cfg = DATASET_CONFIGS.get(dataset_name.lower())
    if not cfg:
        raise ValueError(f"Dataset {dataset_name} not configured. Available: {list(DATASET_CONFIGS.keys())}")

    print(f"\n{'='*80}")
    print(f"STARTING EXPERIMENT | Dataset: {dataset_name} | MagPure 128D")
    print(f"Config: {cfg}")
    print(f"{'='*80}")

    # 2. 加载数据
    edge_list, labels, num_nodes = load_data(dataset_name, data_dir)
    
    # 结果容器
    results = []

    # 3. 多种子循环 (5 Runs)
    for seed in SEEDS:
        print(f"\n>>> Running Seed {seed}...")
        seed_everything(seed)
        
        # --- (A) 数据划分 (Strict 8:1:1) ---
        indices = np.arange(len(edge_list))
        # 先分出 20% (Temp)
        tr_idx, temp_idx = train_test_split(
            indices, test_size=0.2, random_state=seed, stratify=labels
        )
        # 再将 20% 对半分为 Val(10%) 和 Test(10%)
        val_idx, te_idx = train_test_split(
            temp_idx, test_size=0.5, random_state=seed, stratify=labels[temp_idx]
        )
        
        # --- (B) 磁性特征求解 (Unsupervised) ---
        solver = MagneticHadamardSolver(edge_list, num_nodes, q=MAGNETIC_Q)
        fused_embeddings = solver.solve(
            k=EMBEDDING_DIM, 
            tol=cfg['tol'], 
            maxiter=cfg['max_iter']
        )
        del solver; gc.collect()
        
        # --- (C) 特征预处理 ---
        # 划分
        X_train = fused_embeddings[tr_idx]
        y_train = labels[tr_idx]
        X_val = fused_embeddings[val_idx]
        y_val = labels[val_idx]
        X_test = fused_embeddings[te_idx]
        y_test = labels[te_idx]
        
        # 标准化 (Fit on Train, Transform on Val/Test)
        scaler = StandardScaler()
        X_train = scaler.fit_transform(X_train)
        X_val = scaler.transform(X_val)
        X_test = scaler.transform(X_test)
        
        # 标签处理 (确保二分类是 0/1)
        # Amazon (5类) 不需要转换，直接使用 0-4
        if cfg['num_classes'] == 1:
            # 如果标签包含 -1，则统一转为 0/1: (-1, 1) -> (0, 1)
            if np.min(y_train) < 0:
                y_train = (y_train + 1) // 2
                y_val = (y_val + 1) // 2
                y_test = (y_test + 1) // 2
        
        # --- (D) 分类器训练 ---
        model = ClassifierModel(
            hidden_dim=HIDDEN_DIM, 
            dropout_rate=DROPOUT_RATE, 
            num_classes=cfg['num_classes']
        )
        
        # 损失函数选择
        loss_fn = 'sparse_categorical_crossentropy' if cfg['num_classes'] > 1 else 'binary_crossentropy'
        
        model.compile(
            optimizer=keras.optimizers.Adam(LEARNING_RATE),
            loss=loss_fn,
            metrics=['accuracy']
        )
        
        early_stop = keras.callbacks.EarlyStopping(
            monitor='val_loss', 
            patience=PATIENCE, 
            restore_best_weights=True,
            verbose=0
        )
        
        print(f" [Train] Model Training (Loss: {loss_fn})...")
        model.fit(
            X_train, y_train,
            validation_data=(X_val, y_val),
            epochs=TOTAL_EPOCHS,
            batch_size=BATCH_SIZE,
            callbacks=[early_stop],
            verbose=0 
        )
        
        # --- (E) 预测与评估 ---
        # 获取预测概率
        test_probs = model.predict(X_test, verbose=0)
        val_probs = model.predict(X_val, verbose=0) # 仅二分类需要用于阈值搜索
        
        if cfg['num_classes'] > 1:
            # 多分类评估 (Amazon)
            metrics = evaluate_multiclass_predictions(y_test, test_probs, cfg['num_classes'])
        else:
            # 二分类评估 (Flatten probs to 1D array)
            metrics = evaluate_binary_predictions(
                y_test, test_probs.flatten(), 
                y_val, val_probs.flatten()
            )
            
        metrics['seed'] = seed
        results.append(metrics)
        print(f" [Result] Seed {seed} | ACC: {metrics['accuracy']:.4f} | F1-Macro: {metrics['f1_macro']:.4f}")
        
        # 清理内存
        del model, fused_embeddings, X_train, X_val, X_test
        tf.keras.backend.clear_session()
        gc.collect()

    # 4. 汇总结果
    df = pd.DataFrame(results)
    
    print("\n" + "="*80)
    print(f"FINAL AGGREGATED RESULTS | {dataset_name.upper()} | (Mean ± Std)")
    print("-" * 80)
    # 排除 seed 列，计算均值和标准差
    summary = df.drop(columns=['seed']).agg(['mean', 'std']).T
    print(summary.to_string(float_format="%.4f"))
    print("="*80)
    
    # 可选：保存结果到文件
    # df.to_csv(f"results_{dataset_name}.csv", index=False)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Mag2Edge Experiment Runner")
    parser.add_argument('--dataset', type=str, required=True, 
                        choices=['reddit', 'wikiconflict', 'amazon', 'mooc', 'epinions'],
                        help='Name of the dataset to run.')
    parser.add_argument('--data_dir', type=str, default='dataset', 
                        help='Path to the dataset directory.')
    parser.add_argument('--gpu', type=str, default='0', 
                        help='GPU ID to use (e.g., "0" or "1").')
    
    args = parser.parse_args()
    
    try:
        run_experiment(args.dataset, args.data_dir, args.gpu)
    except KeyboardInterrupt:
        print("\n[System] Interrupted by user.")
    except Exception as e:
        print(f"\n[System] Error: {e}")