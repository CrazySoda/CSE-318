import numpy as np
from collections import defaultdict

#helper for sigma calc
#sigma:checks the maximum weight it can contribute to the max cut evaluation
def sigma(v,Y,graph):
    wx=sum(w for neighbour, w in graph[v] if neighbour in Y)
    return wx

#Randomized-1 heuristics
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


#greedy-1 heuristics
def greedy_1(edges,graph):
    X,Y=set(),set()
    
    max_edge = max(edges, key=lambda x: x[2])
    
    u,v,w=max_edge
    X.add(u)
    Y.add(v)
    
    #vertices-takes all the keys of the dict and keeps them in a set
    V = set(graph.keys())
    U=V.difference({u,v})
    
    for z in U:
        #calculate wx
        wx=sum(w for neighbour, w in graph[z] if neighbour in Y)
        wy=sum(w for neighbour, w in graph[z] if neighbour in X)
        if (wx>=wy):
            X.add(z)
        else:
            Y.add(z)
    
    cut_weight=sum(w for u,v,w in edges if (u in X and v in Y) or (u in Y and v in X))
    
    return X,Y,cut_weight



#Semi-Greedy-1
def semi_greedy_1(edges,graph,alpha):
    V=set(graph.keys())
    X,Y=set(),set()
    V_prime=V.copy()
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
        
    cut_weight=sum(w for u,v,w in edges if (u in X and v in Y) or (u in Y and v in X))
    return X,Y,cut_weight

#Local_Search
def local_search(S, S_bar, edges, graph):
    improved = True
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
    
    # Calculate final cut weight
    cut_weight = sum(w for u, v, w in edges if (u in S and v in S_bar) or (u in S_bar and v in S))
    return S, S_bar, cut_weight

def grasp_max_cut(edges, graph, alpha, max_iter):
    best_cut = 0
    best_S, best_S_bar = set(), set()
    
    for i in range(max_iter):
        # Construction phase (semi-greedy)
        S, S_bar, current_cut = semi_greedy_1(edges, graph, alpha)
        #print("Construction:",S,S_bar,current_cut)
        
        # Local search phase
        S, S_bar, improved_cut = local_search(S, S_bar, edges, graph)
        #print("Local:",S,S_bar,improved_cut)
        # Update best solution
        if improved_cut > best_cut:
            best_cut = improved_cut
            best_S, best_S_bar = S.copy(), S_bar.copy()
    
    return best_S, best_S_bar, best_cut

#Testing:
"""edges = [(0, 1, 10), (1, 2, 20), (0, 2, 5), (2, 3, 15)]
graph = defaultdict(list)
for u, v, w in edges:
    graph[u].append((v, w))
    graph[v].append((u, w))
    
print(randomized_1(100,edges,graph))
print(greedy_1(edges,graph))
print(semi_greedy_1(edges,graph,1))
print("GRASP:", grasp_max_cut(edges, graph, 0.5, 5))"""