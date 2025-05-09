import torch
from collections import defaultdict

def read_graphs(file_path, device='cuda'):
    with open(file_path,'r') as f:
        lines=f.readlines()
        
    n,m=map(int,lines[0].split())
    
    edges=[]
    graph=defaultdict(list)
    for line in lines[1:m+1]:
        parts = line.split()
        u = int(parts[0])
        v = int(parts[1])
        w = int(parts[2]) 
        edges.append((u,v,w))
        graph[u].append((v,w))
        graph[v].append((u,w))
    
    # Convert graph to adjacency matrix on GPU
    adj_matrix = torch.zeros((n, n), dtype=torch.float32, device=device)
    for u, v, w in edges:
        adj_matrix[u, v] = w
        adj_matrix[v, u] = w
        
    return n, edges, graph, adj_matrix