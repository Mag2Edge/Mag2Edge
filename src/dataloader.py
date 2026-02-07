# src/dataloader.py
import os
import gzip
import pickle
import urllib.request
import numpy as np
import pandas as pd

def load_data(dataset_name, dataset_dir="dataset"):
    """
    统一的数据加载入口。
    
    Args:
        dataset_name: 数据集名称 (reddit, wikiconflict, amazon, mooc, epinions)
        dataset_dir: 数据集根目录
        
    Returns:
        edge_list: (num_edges, 2) int32 数组，包含源节点和目标节点索引
        labels: (num_edges, ) int32 数组，包含边标签
        num_nodes: 节点总数
    """
    dataset_name = dataset_name.lower()
    
    if dataset_name == 'reddit':
        return load_reddit_data(dataset_dir)
    elif dataset_name == 'wikiconflict':
        return load_wikiconflict_data(dataset_dir)
    elif dataset_name == 'amazon':
        return load_amazon_data(dataset_dir)
    elif dataset_name == 'mooc':
        return load_mooc_data(dataset_dir)
    elif dataset_name == 'epinions':
        return load_epinions_data(dataset_dir)
    else:
        raise ValueError(f"Unknown dataset: {dataset_name}")

# -----------------------------------------------------------------------------
# 1. Reddit Data Loader
# -----------------------------------------------------------------------------
def load_reddit_data(dataset_dir):
    print("Loading Reddit dataset...")
    files = {
        'train_edge': os.path.join(dataset_dir, "train_edges.csv"),
        'val_edge': os.path.join(dataset_dir, "val_edges.csv"),
        'test_edge': os.path.join(dataset_dir, "test_edges.csv"),
        'train_lbl': os.path.join(dataset_dir, "train_embeds_labels.pkl"),
        'val_lbl': os.path.join(dataset_dir, "val_embeds_labels.pkl"),
        'test_lbl': os.path.join(dataset_dir, "test_embeds_labels.pkl")
    }

    # 检查文件
    for k, v in files.items():
        if not os.path.exists(v):
            raise FileNotFoundError(f"Missing file: {v}")

    # 读取边和标签
    train_df = pd.read_csv(files['train_edge'])
    val_df = pd.read_csv(files['val_edge'])
    test_df = pd.read_csv(files['test_edge'])

    with open(files['train_lbl'], "rb") as f: _, tl = pickle.load(f)
    with open(files['val_lbl'], "rb") as f: _, vl = pickle.load(f)
    with open(files['test_lbl'], "rb") as f: _, tel = pickle.load(f)

    # 标签预处理 (确保是 -1/1)
    tl = np.where(tl > 0, 1, -1).flatten()
    vl = np.where(vl > 0, 1, -1).flatten()
    tel = np.where(tel > 0, 1, -1).flatten()

    # 合并所有数据 (按照原始脚本逻辑，先合并再在 main 中重新 split)
    raw_edges = []
    raw_edges.extend(list(zip(train_df['SENDER_id'].values, train_df['RECEIVER_id'].values, tl)))
    raw_edges.extend(list(zip(val_df['SENDER_id'].values, val_df['RECEIVER_id'].values, vl)))
    raw_edges.extend(list(zip(test_df['SENDER_id'].values, test_df['RECEIVER_id'].values, tel)))

    # 节点映射
    unique_nodes = sorted(set([e[0] for e in raw_edges] + [e[1] for e in raw_edges]))
    node_to_idx = {node: idx for idx, node in enumerate(unique_nodes)}

    edge_list = np.array([[node_to_idx[e[0]], node_to_idx[e[1]]] for e in raw_edges], dtype=np.int32)
    labels = np.array([e[2] for e in raw_edges], dtype=np.int32)

    print(f"Loaded Reddit: {len(unique_nodes)} Nodes, {len(edge_list)} Edges.")
    return edge_list, labels, len(unique_nodes)

# -----------------------------------------------------------------------------
# 2. WikiConflict Data Loader
# -----------------------------------------------------------------------------
def load_wikiconflict_data(dataset_dir):
    print("Loading WikiConflict dataset...")
    dataset_file = os.path.join(dataset_dir, "wikiconflict.csv")
    
    if not os.path.exists(dataset_file):
        raise FileNotFoundError(f"Dataset file not found: {dataset_file}")

    try:
        # Assuming CSV without header: source, target, label
        df = pd.read_csv(dataset_file, header=None, names=['source', 'target', 'label'])
    except Exception as e:
        raise RuntimeError(f"Failed to read CSV file: {e}")

    # Build Edge List (Directed)
    edges_raw = df[['source', 'target', 'label']].values.astype(np.int32)

    # Node Re-indexing (Compact Indexing)
    unique_nodes = np.unique(edges_raw[:, :2])
    node_to_idx = {node: idx for idx, node in enumerate(unique_nodes)}

    u = np.vectorize(node_to_idx.get)(edges_raw[:, 0])
    v = np.vectorize(node_to_idx.get)(edges_raw[:, 1])
    labels = edges_raw[:, 2]

    edge_list = np.stack([u, v], axis=1).astype(np.int32)
    print(f"Loaded WikiConflict: {len(unique_nodes)} Nodes, {len(edge_list)} Edges.")
    
    return edge_list, labels, len(unique_nodes)

# -----------------------------------------------------------------------------
# 3. Amazon Movies Data Loader
# -----------------------------------------------------------------------------
def download_amazon_movie_data(dataset_dir):
    os.makedirs(dataset_dir, exist_ok=True)
    movie_file = os.path.join(dataset_dir, "movies.txt.gz")
    if os.path.exists(movie_file): 
        return movie_file
    
    print("Downloading Amazon Movie dataset...")
    url = "https://snap.stanford.edu/data/movies.txt.gz"
    urllib.request.urlretrieve(url, movie_file)
    return movie_file

def load_amazon_data(dataset_dir):
    movie_file = download_amazon_movie_data(dataset_dir)
    edges = []
    current_review = {}

    print(f" [Data] Parsing Amazon dataset...")
    with gzip.open(movie_file, 'rt', encoding='latin-1') as f:
        for line in f:
            line = line.strip()
            if not line:
                if 'u' in current_review and 'p' in current_review and 's' in current_review:
                    # 映射为 0-4
                    score = int(max(1, min(5, current_review['s']))) - 1
                    edges.append((current_review['u'], current_review['p'], score))
                current_review = {}
                continue

            if ':' in line:
                parts = line.split(':', 1)
                key, val = parts[0].strip(), parts[1].strip()
                if key == 'review/userId': current_review['u'] = val
                elif key == 'product/productId': current_review['p'] = val
                elif key == 'review/score': current_review['s'] = float(val)

    edges_df = pd.DataFrame(edges, columns=['u', 'v', 'label'])
    unique_nodes = np.unique(np.concatenate([edges_df['u'], edges_df['v']]))
    node_to_idx = {node: idx for idx, node in enumerate(unique_nodes)}

    u_idx = edges_df['u'].map(node_to_idx).values
    v_idx = edges_df['v'].map(node_to_idx).values

    edge_list = np.stack([u_idx, v_idx], axis=1)
    labels = edges_df['label'].values
    
    print(f"Loaded Amazon: {len(unique_nodes)} Nodes, {len(edge_list)} Edges.")
    return edge_list, labels, len(unique_nodes)

# -----------------------------------------------------------------------------
# 4. MOOC Data Loader
# -----------------------------------------------------------------------------
def load_mooc_data(dataset_dir):
    print("Loading MOOC dataset...")
    actions_file = os.path.join(dataset_dir, "mooc_actions.tsv")
    labels_file = os.path.join(dataset_dir, "mooc_action_labels.tsv")

    if not os.path.exists(actions_file) or not os.path.exists(labels_file):
        raise FileNotFoundError(f"Missing MOOC files in {dataset_dir}")

    actions_df = pd.read_csv(actions_file, sep='\t')
    labels_df = pd.read_csv(labels_file, sep='\t')
    
    merged_df = pd.merge(actions_df, labels_df, on='ACTIONID')
    edges_raw = merged_df[['USERID', 'TARGETID', 'LABEL']].values

    unique_nodes = np.unique(edges_raw[:, :2])
    node_to_idx = {node: idx for idx, node in enumerate(unique_nodes)}
    num_nodes = len(unique_nodes)

    u = np.vectorize(node_to_idx.get)(edges_raw[:, 0])
    v = np.vectorize(node_to_idx.get)(edges_raw[:, 1])
    labels = edges_raw[:, 2].astype(np.int32)

    edge_list = np.stack([u, v], axis=1)
    print(f"Loaded MOOC: {num_nodes} Nodes, {len(edge_list)} Edges.")
    return edge_list, labels, num_nodes

# -----------------------------------------------------------------------------
# 5. Epinions Data Loader
# -----------------------------------------------------------------------------
def load_epinions_data(dataset_dir):
    print("Loading Epinions dataset...")
    dataset_file = os.path.join(dataset_dir, "epinions_data.txt")
    
    if not os.path.exists(dataset_file):
        raise FileNotFoundError(f"Missing: {dataset_file}")

    edges_raw = []
    with open(dataset_file, 'r') as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) >= 3:
                u, v, rating = int(parts[0]), int(parts[1]), int(parts[2])
                label = 1 if rating > 0 else -1
                edges_raw.append([u, v, label])

    edges_np = np.array(edges_raw, dtype=np.int32)
    unique_nodes = np.unique(edges_np[:, :2])
    node_to_idx = {node: idx for idx, node in enumerate(unique_nodes)}
    num_nodes = len(unique_nodes)

    u = np.vectorize(node_to_idx.get)(edges_np[:, 0])
    v = np.vectorize(node_to_idx.get)(edges_np[:, 1])
    labels = edges_np[:, 2].astype(np.int32)
    
    edge_list = np.stack([u, v], axis=1)
    print(f"Loaded Epinions: {num_nodes} Nodes, {len(edge_list)} Edges.")
    return edge_list, labels, num_nodes