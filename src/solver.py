# src/solver.py
import numpy as np
import scipy.sparse as sp
from scipy.sparse.linalg import eigsh, LinearOperator

class MagneticHadamardSolver:
    """
    Reddit Dataset | Magnetic Laplacian Pure Decomposition | 128D Strict
    WikiConflict / Amazon / Mooc / Epinions 通用求解器。
    
    [cite_start]修改说明[cite: 1, 19, 39, 54, 385]:
    1. 移除了泰勒展开谱放大器 (Taylor Expansion Spectral Amplifier)。
    2. 仅保留磁拉普拉斯算子的分解与双线性哈达玛特征融合 (Bilinear Hadamard Fusion)。
    """
    
    def __init__(self, edge_list, num_nodes, q=0.25):
        """
        初始化求解器参数。
        
        Args:
            edge_list: 边的列表 (u, v) 数组。
            num_nodes: 节点总数。
            [cite_start]q: 磁偏角参数 (捕捉方向流)，默认 0.25 [cite: 3, 21, 41, 55, 387]。
        """
        self.num_nodes = num_nodes
        self.num_edges = len(edge_list)
        self.q = q
        
        # 提取 u, v
        self.u = edge_list[:, 0]
        self.v = edge_list[:, 1]
        
        # 预构建复数关联矩阵 E_q
        self._build_incidence_matrix()

    def _build_incidence_matrix(self):
        """
        构建 Hermitian Incidence Matrix E_q。
        [cite_start]Source = 1, Target = -exp(i * 2pi * q) [cite: 8, 25, 45, 59, 390]。
        """
        # 磁性相位
        phase = np.exp(1j * 2 * np.pi * self.q)
        
        # 构造 CSR 矩阵所需的 row, col, data
        row = np.concatenate([self.u, self.v])
        e_ids = np.arange(self.num_edges)
        col = np.concatenate([e_ids, e_ids])
        
        # Hermitian Incidence: source=1, target=-phase
        data = np.concatenate([
            np.ones(self.num_edges, dtype=np.complex64),
            -phase * np.ones(self.num_edges, dtype=np.complex64)
        ])
        
        self.E_q = sp.csr_matrix(
            (data, (row, col)), 
            shape=(self.num_nodes, self.num_edges),
            dtype=np.complex64
        )

    def solve(self, k=128, tol=1e-8, maxiter=5000):
        """
        求解磁性特征向量并进行双线性哈达玛融合。
        
        Args:
            [cite_start]k: 嵌入维度，默认 128 [cite: 3]。
            tol: eigsh 收敛容差 (不同数据集可能不同，需从外部传入)。
            maxiter: eigsh 最大迭代次数。
            
        Returns:
            fused_embeddings: (num_edges, k) 的实数特征矩阵。
        """
        print(f" [Solver] Computing Magnetic Eigenvectors (Pure Decomposition) k={k}...")
        
        # 定义隐式算子 L_op = E_q^H @ E_q
        # 避免显式构建 O(m^2) 的拉普拉斯矩阵
        def matvec(x):
            return self.E_q.conj().T @ (self.E_q @ x)
            
        L_op = LinearOperator(
            shape=(self.num_edges, self.num_edges), 
            matvec=matvec,
            dtype=np.complex64
        )
        
        # 1. 求解 Hermitian 矩阵特征对
        # [cite_start]which='LM' (Largest Magnitude) 对应找梯度流 [cite: 9, 26, 46, 60, 391]
        vals, vecs = eigsh(L_op, k=k, which='LM', tol=tol, maxiter=maxiter)
        
        # --- 修改点：移除泰勒展开谱放大器 ---
        # 原逻辑中计算 amp_vals 并注入能量的部分已移除
        
        # [cite_start]2. 双线性哈达玛积融合 [cite: 10, 27, 47, 60, 391]
        
        # (1) 提取实虚部
        real_part = np.real(vecs)
        imag_part = np.imag(vecs)
        
        # (2) 计算交互项 (Element-wise Product)
        # 捕捉 Real 和 Imag 的非线性耦合
        interaction = np.multiply(real_part, imag_part)
        
        # (3) 残差融合: Interaction + Real + Imag
        fused_embeddings = interaction + real_part + imag_part
        
        # --- 修改点：移除能量注入 ---
        # 原逻辑：fused_embeddings = fused_embeddings * amp_vals.reshape(1, -1) 被移除
        
        # 3. 排序 (按特征值大小降序)
        idx = np.argsort(vals)[::-1]
        
        # 返回严格的 k 维实数特征
        return fused_embeddings[:, idx].astype(np.float32)