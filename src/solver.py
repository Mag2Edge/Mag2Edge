import numpy as np
import scipy.sparse as sp
from scipy.sparse.linalg import eigsh, LinearOperator

class MagneticHadamardSolver:
   
    
    def __init__(self, edge_list, num_nodes, q=0.25):
      
        self.num_nodes = num_nodes
        self.num_edges = len(edge_list)
        self.q = q
        
        self.u = edge_list[:, 0]
        self.v = edge_list[:, 1]
        
        self._build_incidence_matrix()

    def _build_incidence_matrix(self):
   
        phase = np.exp(1j * 2 * np.pi * self.q)
        

        row = np.concatenate([self.u, self.v])
        e_ids = np.arange(self.num_edges)
        col = np.concatenate([e_ids, e_ids])
        

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
       
        print(f" [Solver] Computing Magnetic Eigenvectors (Pure Decomposition) k={k}...")
        

        def matvec(x):
            return self.E_q.conj().T @ (self.E_q @ x)
            
        L_op = LinearOperator(
            shape=(self.num_edges, self.num_edges), 
            matvec=matvec,
            dtype=np.complex64
        )
        
       
        vals, vecs = eigsh(L_op, k=k, which='LM', tol=tol, maxiter=maxiter)
        

        real_part = np.real(vecs)
        imag_part = np.imag(vecs)
        
      
        interaction = np.multiply(real_part, imag_part)
        
     
        fused_embeddings = interaction + real_part + imag_part
        

        
        idx = np.argsort(vals)[::-1]
        
     
        return fused_embeddings[:, idx].astype(np.float32)
