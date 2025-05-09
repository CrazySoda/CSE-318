import numpy as np
from collections import defaultdict
import torch

# GPU-optimized sigma calculation
def sigma_gpu(v, Y_mask, adj_matrix):
    """Y_mask should be a boolean tensor on GPU"""
    return adj_matrix[v] @ Y_mask.float()

# Modified randomized_1 (no GPU needed here)
def randomized_1(n,edges,graph):
    totalCutWeight=0
    
    for i in range(n):
        X,Y=set(),set()
        V = set(graph.keys())
        assignments = np.random.choice([0,1],size=len(graph),p=[0.5,0.5])
        X={int(v) for v , assign in zip(graph.keys(),assignments)if assign==1}
        Y=V-X    
        cut_weight=sum(w for u,v,w in edges if(u in X and v in Y)or(u in Y and v in X))
        totalCutWeight+=cut_weight
    
    averageCutWeight = totalCutWeight/n
    return averageCutWeight

# Modified greedy_1 with GPU support
def greedy_1(edges, graph, adj_matrix=None):
    X,Y = set(),set()
    max_edge = max(edges, key=lambda x: x[2])
    u,v,w = max_edge
    X.add(u)
    Y.add(v)
    
    V = set(graph.keys())
    U = V.difference({u,v})
    
    if adj_matrix is not None:
        # GPU-accelerated version
        X_mask = torch.zeros(adj_matrix.size(0), dtype=torch.bool, device='cuda')
        Y_mask = torch.zeros(adj_matrix.size(0), dtype=torch.bool, device='cuda')
        X_mask[u], Y_mask[v] = True, True
        
        for z in U:
            wx = (adj_matrix[z] * Y_mask.float()).sum()
            wy = (adj_matrix[z] * X_mask.float()).sum()
            if wx >= wy:
                X.add(z)
                X_mask[z] = True
            else:
                Y.add(z)
                Y_mask[z] = True
    else:
        # Original CPU version
        for z in U:
            wx = sum(w for neighbour, w in graph[z] if neighbour in Y)
            wy = sum(w for neighbour, w in graph[z] if neighbour in X)
            if wx >= wy:
                X.add(z)
            else:
                Y.add(z)
    
    cut_weight = sum(w for u,v,w in edges if (u in X and v in Y) or (u in Y and v in X))
    return X, Y, cut_weight

# GPU-optimized semi_greedy_1
def semi_greedy_1(edges, graph, alpha, adj_matrix=None):
    V = set(graph.keys())
    X,Y = set(),set()
    V_prime = V.copy()
    
    if adj_matrix is not None:
        # GPU version
        X_mask = torch.zeros(adj_matrix.size(0), dtype=torch.bool, device='cuda')
        Y_mask = torch.zeros(adj_matrix.size(0), dtype=torch.bool, device='cuda')
        
        while V_prime:
            sigma_x = torch.zeros(len(V), device='cuda')
            sigma_y = torch.zeros(len(V), device='cuda')
            
            for v in V_prime:
                sigma_x[v] = sigma_gpu(v, Y_mask, adj_matrix)
                sigma_y[v] = sigma_gpu(v, X_mask, adj_matrix)
            
            greedy_vals = torch.max(sigma_x, sigma_y)
            min_vals = torch.min(sigma_x, sigma_y)
            
            w_min = min_vals.min()
            w_max = greedy_vals.max()
            mu = w_min + alpha*(w_max - w_min)
            
            RCL = [v for v in V_prime if greedy_vals[v] >= mu]
            if not RCL:
                RCL = list(V_prime)
                
            v = int(np.random.choice(RCL))
            
            if sigma_x[v] >= sigma_y[v]:
                X.add(v)
                X_mask[v] = True
            else:
                Y.add(v)
                Y_mask[v] = True
                
            V_prime.remove(v)
    else:
        # Original CPU version
        while V_prime:
        #dictionary of sigmas
            sigma_x={v:sigma(v,Y,graph) for v in V_prime}
            sigma_y={v:sigma(v,X,graph) for v in V_prime}
            
            greedy_values={v:max(sigma_x[v],sigma_y[v]) for v in V_prime}
            
            min_values={v:min(sigma_x[v],sigma_y[v]) for v in V_prime}
            
            w_min= min(min_values.values())
            w_max= max(greedy_values.values())
            
            #alpha:if 1;then best move as only the greediest values will be available. if 0; then fully random as all nodes will qualify
            mu = w_min + alpha*(w_max - w_min)
            RCL = [
                v for v in V_prime
                if greedy_values[v] >= mu
            ]
            
            if not RCL:
                RCL = list(V_prime)
            
            v = int(np.random.choice(RCL))
            
            if sigma_x[v] >= sigma_y[v]:
                X.add(v)
            else:
                Y.add(v)
                
            V_prime.remove(v)
    
    cut_weight = sum(w for u,v,w in edges if (u in X and v in Y) or (u in Y and v in X))
    return X, Y, cut_weight

# GPU-optimized local search
def local_search(S, S_bar, edges, graph, adj_matrix=None):
    improved = True
    
    if adj_matrix is not None:
        # GPU version
        S_mask = torch.zeros(adj_matrix.size(0), dtype=torch.bool, device='cuda')
        S_bar_mask = torch.zeros(adj_matrix.size(0), dtype=torch.bool, device='cuda')
        for v in S: S_mask[v] = True
        for v in S_bar: S_bar_mask[v] = True
        
        while improved:
            improved = False
            best_delta = 0
            best_vertex = None
            
            for v in list(S) + list(S_bar):
                if v in S:
                    delta = (sigma_gpu(v, S_mask, adj_matrix) - 
                            sigma_gpu(v, S_bar_mask, adj_matrix))
                else:
                    delta = (sigma_gpu(v, S_bar_mask, adj_matrix) - 
                            sigma_gpu(v, S_mask, adj_matrix))
                
                if delta > best_delta:
                    best_delta = delta
                    best_vertex = v
            
            if best_delta > 0:
                improved = True
                if best_vertex in S:
                    S.remove(best_vertex)
                    S_bar.add(best_vertex)
                    S_mask[best_vertex] = False
                    S_bar_mask[best_vertex] = True
                else:
                    S_bar.remove(best_vertex)
                    S.add(best_vertex)
                    S_bar_mask[best_vertex] = False
                    S_mask[best_vertex] = True
    else:
        while improved:
            improved = False
            best_delta = 0
            best_vertex = None
            
            # Check all vertices for possible moves
            for v in set(S) | set(S_bar):    
                
                if v in S:
                    #delta:gains all the connections of S set and loses all connections with S_bar
                    delta = sigma(v,S,graph) - sigma(v,S_bar,graph)  # Gain if moved to S_bar
                else: 
                    delta = sigma(v,S_bar,graph) - sigma(v,S,graph)  # Gain if moved to S
                
                # Track the best improving move
                if delta > best_delta:
                    best_delta = delta
                    best_vertex = v
            
            # Apply the best move if it improves the cut
            if best_delta > 0:
                if best_vertex in S:
                    S.remove(best_vertex)
                    S_bar.add(best_vertex)
                else:
                    S_bar.remove(best_vertex)
                    S.add(best_vertex)
                improved = True
        
    cut_weight = sum(w for u,v,w in edges if (u in S and v in S_bar) or (u in S_bar and v in S))
    return S, S_bar, cut_weight

def grasp_max_cut(edges, graph, alpha, max_iter, adj_matrix=None):
    best_cut = 0
    best_S, best_S_bar = set(), set()
    
    for _ in range(max_iter):
        S, S_bar, current_cut = semi_greedy_1(edges, graph, alpha, adj_matrix)
        S, S_bar, improved_cut = local_search(S, S_bar, edges, graph, adj_matrix)
        
        if improved_cut > best_cut:
            best_cut = improved_cut
            best_S, best_S_bar = S.copy(), S_bar.copy()
    
    return best_S, best_S_bar, best_cut