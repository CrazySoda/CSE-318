from collections import defaultdict

def read_graphs(file_path):
    with open(file_path,'r') as f:
        lines=f.readlines()
        
    n,m=map(int,lines[0].split())
    
    edges=[]
    for line in lines[1:m+1]:
        parts = line.split()
        u = int(parts[0])
        v = int(parts[1])
        w = int(parts[2]) 
        edges.append((u,v,w))
    
    graph=defaultdict(list)
    for u,v,w in edges:
        graph[u].append((v,w))
        graph[v].append((u,w))
        
    return n,edges,graph
        