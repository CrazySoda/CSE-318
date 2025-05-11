#include <bits/stdc++.h>
#include <execution>
#include <filesystem>
using namespace std;
using namespace chrono;
namespace fs = std::filesystem;

// ─────────────────────────── Graph ──────────────────────────────────────────
class FastGraph
{
    unordered_map<int, unordered_map<int, int>> adj; // w(u,v)
    vector<tuple<int, int, int>> edges;              // (u,v,w)
    int vertexCount=0;                              
public:
    void add_edge(int u, int v, int w)
    {
        if (adj[u].empty()) vertexCount++;
        if (adj[v].empty()) vertexCount++;
        adj[u][v] = w;
        adj[v][u] = w;
        edges.emplace_back(u, v, w);
    }
    int n() const { return vertexCount; } 
    size_t m() const { return edges.size(); }
    const auto &E() const { return edges; }
    const auto &neighbours(int u) const { return adj.at(u); } 
    vector<int> vertices() const
    { 
        vector<int> v;
        //reserve the memory upto the number of vertices to make it memory efficient
        v.reserve(adj.size());
        for (auto &p : adj)
            v.push_back(p.first);
        sort(v.begin(), v.end());
        return v;
    }
};

// ─────────────────── Random Probability Generator ────────────────────────────────
thread_local mt19937_64 rng(steady_clock::now().time_since_epoch().count());
static uniform_real_distribution<double> prob01(0.0, 1.0);

// ─────────────────────── Max‑Cut Solver ─────────────────────────────────────
class MaxCutSolver
{
    const FastGraph &g;

public:
    explicit MaxCutSolver(const FastGraph &G) : g(G) {}

    // cut value given one side S (other is its complement wrt present vertices)
    double cut(const unordered_set<int> &S) const
    {
        double val = 0;
        for (auto [u, v, w] : g.E())
            //xor operation either u in S or v in S
            if ((S.count(u) > 0) ^ (S.count(v) > 0))
                val += w;
        return val;
    }

    // ---------------- Randomized ----------------
    double randomized(int trials)
    {
        vector<int> V = g.vertices();
        double total = 0;

#pragma omp parallel for reduction(+ : total)
        for (int t = 0; t < trials; ++t)
        {
            unordered_set<int> S;
            for (int v : V)
                if (prob01(rng) >= 0.5)
                    S.insert(v);
            total += cut(S);
        }
        return total / trials;
    }

    // ---------------- Greedy ----------------
    tuple<unordered_set<int>, unordered_set<int>, double> greedy()
    {
        auto maxE = *max_element(g.E().begin(), g.E().end(), [](auto &a, auto &b)
                                 { return get<2>(a) < get<2>(b); });
        unordered_set<int> X{get<0>(maxE)}, Y{get<1>(maxE)};
        for (int v : g.vertices())
            if (!X.count(v) && !Y.count(v))
            {
                long long wx = 0, wy = 0;
                for (auto [nbr, w] : g.neighbours(v))
                {
                    if (Y.count(nbr))
                        wx += w;
                    if (X.count(nbr))
                        wy += w;
                }
                if (wx>=wy)
                    X.insert(v);
                else
                    Y.insert(v);
            }
        return {X, Y, cut(X)};
    }

    // ---------------- Semi‑Greedy ----------------
    tuple<unordered_set<int>, unordered_set<int>, double> semi_greedy(double alpha)
    {
        int N = g.n();
        vector<char> active(N + 1, 0);//Not in any sets X or Y
        vector<int> vertices = g.vertices();
        for (int v : vertices)
            active[v] = 1;
        
        unordered_set<int> X, Y;
        int remaining = accumulate(active.begin(), active.end(), 0);
        
        
        vector<double> greedy_value(N + 1, 0.0);

        while (remaining)
        {
            for (int v = 1; v <= N; ++v)
                if (active[v])
                {
                    long long wx = 0, wy = 0;
                    for (auto [nbr, w] : g.neighbours(v))
                    {
                        if (Y.count(nbr))
                            wx += w;
                        if (X.count(nbr))
                            wy += w;
                    }
                    greedy_value[v] = max(wx ,wy);
                }
            // determine RCL threshold
            double w_min = 1e100, w_max = -1e100;
            for (int v = 1; v <= N; ++v)
                if (active[v])
                {
                    w_min = min(w_min, greedy_value[v]);
                    w_max = max(w_max, greedy_value[v]);
                }
            double mu = w_min + alpha * (w_max - w_min);
            vector<int> RCL;
            for (int v = 1; v <= N; ++v)
                if (active[v] && greedy_value[v] >= mu)
                    RCL.push_back(v);
            if (RCL.empty())
                for (int v = 1; v <= N; ++v)
                    if (active[v])
                        RCL.push_back(v);
            int v = RCL[uniform_int_distribution<size_t>(0, RCL.size() - 1)(rng)];

            long long wx = 0, wy = 0;
            for (auto [nbr, w] : g.neighbours(v))
            {
                if (Y.count(nbr))
                    wx += w;
                if (X.count(nbr))
                    wy += w;
            }
            if(wx>=wy)
                X.insert(v);
            else
                Y.insert(v);
            active[v] = 0;
            --remaining;
        }
        return {X, Y, cut(X)};
    }

    // ---------------sigma-------------------------
    long long sigma(int v, const unordered_set<int>& target_set) const
    {
        long long sum = 0;
        for (auto [nbr, w] : g.neighbours(v))
        {
            if (target_set.count(nbr) > 0)
                sum += w;
        }
        return sum;
    }

    // ---------------- Local Search ----------------
    tuple<unordered_set<int>, unordered_set<int>, double> local_search(unordered_set<int> S, unordered_set<int> Sbar)
    {
        bool improved = true;
        
        while(improved)
        {
            improved = false;
            long long bestDelta = 0;
            int bestV = -1;
            
            // Check all vertices for possible moves
            for (int v : g.vertices())
            {
                long long delta = 0;
                
                if (S.count(v) > 0) {
                    // v is in S, calculate gain if moved to Sbar
                    delta = sigma(v, S) - sigma(v, Sbar);
                } else if (Sbar.count(v) > 0) {
                    // v is in Sbar, calculate gain if moved to S
                    delta = sigma(v, Sbar) - sigma(v, S);
                }
                
                if (delta > bestDelta)
                {
                    bestDelta = delta;
                    bestV = v;
                }
            }
            
            if (bestDelta > 0)
            {
                improved = true;
                // Move the vertex to the other set
                if (S.count(bestV) > 0)
                {
                    S.erase(bestV);
                    Sbar.insert(bestV);
                }
                else
                {
                    Sbar.erase(bestV);
                    S.insert(bestV);
                }
            }
        }
        
        double cutValue = cut(S);
        return {S, Sbar, cutValue};
    }

    // ---------------- GRASP ----------------
    double grasp(double alpha, int iters)
    {
        double best = -1e18;
        
#pragma omp parallel
        {
            double local_best = -1e18;
            unordered_set<int> best_S, best_Sbar;
            
#pragma omp for nowait
            for (int i = 0; i < iters; ++i)
            {
                unordered_set<int> S, Sbar;
                double val;
                
                // Construction phase: semi-greedy
                tie(S, Sbar, val) = semi_greedy(alpha);
                
                // Improvement phase: local search
                tie(S, Sbar, val) = local_search(S, Sbar);
                
                // Update local best solution
                if (val > local_best)
                {
                    local_best = val;
                    best_S = S;
                    best_Sbar = Sbar;
                }
            }
            
#pragma omp critical
            {
                if (local_best > best)
                {
                    best = local_best;
                }
            }
        }
        
        return best;
    }
};

// ─────────────────────── File Reader ───────────────────────────────────────
FastGraph read_graph(const string &file)
{
    FastGraph g;
    ifstream in(file);
    int n, m, u, v, w;
    if (!(in >> n >> m))
        return g;
    for (int i = 0; i < m && (in >> u >> v >> w); ++i)
        g.add_edge(u, v, w);
    return g;
}

// ───────────────────────── main ────────────────────────────────────────────
int main()
{
    const string DIR = "../../graph_GRASP/set1", CSV = "results.csv";
    const int TR = 100, LS = 5;
    const int GR[] = {50,100,200,300};
    const double A = 0.5;
    vector<pair<int, string>> files;
    for (auto &e : fs::directory_iterator(DIR))
        if (e.is_regular_file())
        {
            string ext = e.path().extension().string(), stem = e.path().stem().string();
            if (ext == ".rud" && stem.size() > 1 && stem[0] == 'g')
                files.emplace_back(stoi(stem.substr(1)), e.path().string());
        }
    sort(files.begin(), files.end());

    ofstream csv(CSV);
    csv << "Problem,,,Constructive algorithm,,,LocalSearch,,GRASP-50,,GRASP-100,,GRASP-200,,GRASP-300,,Known best solution or upper bound\n";
    csv << "name,|V| or n,|E| or m,Simple Randomized or Randomized-1,Simple Greedy or Greedy-1,Semi-greedy-1,Average value,No. of iterations,GRASP-50 Val,Iterations,GRASP-100 Val,Iterations,GRASP-200 Val,Iterations,GRASP-300 Val,Iterations,Best\n";
    
    for (auto [id, path] : files)
    {
        cerr << "\n→ " << path << "\n";
        FastGraph g = read_graph(path);
        MaxCutSolver s(g);
        double r = s.randomized(TR), g1, s1, lsAvg = 0;
        double grasp[]={0.0,0.0,0.0,0.0};
        unordered_set<int> tmp1, tmp2;
        tie(tmp1, tmp2, g1) = s.greedy();
        tie(tmp1, tmp2, s1) = s.semi_greedy(A);
        
        // Run local search multiple times with different starting solutions
        for (int i = 0; i < LS; ++i)
        {
            // Use different alpha values for semi-greedy to get diverse starting solutions
            double a = prob01(rng);
            unordered_set<int> S, T;
            double val;
            
            // Construction phase
            tie(S, T, val) = s.semi_greedy(a);
            
            // Local search improvement phase 
            tie(S, T, val) = s.local_search(S, T);
            lsAvg += val;
        }
        lsAvg /= LS;
        
        // Run GRASP with different iteration counts
        for(int i=0; i<4; i++){
            grasp[i] = s.grasp(A, GR[i]);
            cerr << "GRASP-" << GR[i] << " completed with value: " << grasp[i] << "\n";
        }
        
        double best = max({r, g1, s1, lsAvg, grasp[0], grasp[1], grasp[2], grasp[3]});
        csv << fs::path(path).stem().string() << "," << g.n() << "," << g.m() << "," << r << "," << g1 << "," << s1 << "," << lsAvg << "," << LS << "," << grasp[0] << "," << GR[0] << "," << grasp[1] << "," << GR[1] << "," << grasp[2] << "," << GR[2] << "," << grasp[3] << "," << GR[3] << "," << best << "\n";
    }

    cerr << "\nSaved → " << CSV << "\n";

    return 0;
}