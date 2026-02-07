import os
import random
import numpy as np
import tensorflow as tf
from sklearn.metrics import (
    accuracy_score, f1_score, balanced_accuracy_score,
    roc_auc_score, average_precision_score
)
from sklearn.preprocessing import label_binarize

def seed_everything(seed):

    random.seed(seed)
    np.random.seed(seed)
    tf.random.set_seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)

def config_gpu(gpu_id='0'):
   
    os.environ['CUDA_VISIBLE_DEVICES'] = gpu_id
    os.environ['TF_GPU_ALLOCATOR'] = 'cuda_malloc_async'
    
    gpus = tf.config.experimental.list_physical_devices('GPU')
    if gpus:
        try:
            for gpu in gpus:
                tf.config.experimental.set_memory_growth(gpu, True)
        except RuntimeError as e:
            print(f"[System] GPU Error: {e}")

def evaluate_binary_predictions(y_test, test_probs, y_val, val_probs):

    best_t, best_f1 = 0.5, 0

    for t in np.arange(0.01, 0.99, 0.01):
       
        val_preds = (val_probs >= t).astype(int)
        f = f1_score(y_val, val_preds, average='macro')
        if f > best_f1:
            best_f1, best_t = f, t
            
 
    test_preds = (test_probs >= best_t).astype(int)
    
   
    metrics = {
        'accuracy': accuracy_score(y_test, test_preds),
        'f1_macro': f1_score(y_test, test_preds, average='macro'),
        'balanced_acc': balanced_accuracy_score(y_test, test_preds),
        'auc': roc_auc_score(y_test, test_probs),
        'ap': average_precision_score(y_test, test_probs),
        'optimal_threshold': best_t
    }
    return metrics

def evaluate_multiclass_predictions(y_test, test_probs, num_classes):

   
    preds = np.argmax(test_probs, axis=1)
    
    
    y_test_bin = label_binarize(y_test, classes=range(num_classes))
    

    metrics = {
        'accuracy': accuracy_score(y_test, preds),
        'f1_macro': f1_score(y_test, preds, average='macro'),
        'balanced_acc': balanced_accuracy_score(y_test, preds),

        'auc_ovr': roc_auc_score(y_test, test_probs, multi_class='ovr', average='macro'),
        'mAP': average_precision_score(y_test_bin, test_probs, average='macro')
    }
    return metrics
