import torch
import torch.nn.functional as F
from torch_geometric.data import Data
import networkx as nx
import numpy as np
from collections import defaultdict
from model import GNNPrefetch 

class LocalCache:
    def __init__(self):
        self.cache = defaultdict(dict)  # {node_id: {"content": ..., "status": "cached"}}

    def prefetch(self, node_id, content="Simulated content"):
        self.cache[node_id] = {"content": content, "status": "cached"}

    def get(self, node_id):
        return self.cache.get(node_id, None)

    def show(self):
        if not self.cache:
            print("Cache is empty.")
        else:
            print("📦 Current Cache State:")
            for node, data in self.cache.items():
                print(f"  - Node {node}: {data['status']}")

data = torch.load("E:/Research/R004SE-PrefetchingGnn/user-session-sim/gnn_prefetch_dataset.pt", weights_only=False)
model = GNNPrefetch(data.x.size(1), 128, 64, num_layers=3, model_type="sage")
model.load_state_dict(torch.load("gnn_prefetch_linkpred.pt", weights_only=False))
model.eval()

class PrefetchManager:
    def __init__(self, model, data, cache, top_k=5):
        self.model = model
        self.data = data
        self.cache = cache
        self.top_k = top_k

    def predict_next(self, current_node):
        with torch.no_grad():
            z = self.model.get_embedding(self.data.x, self.data.edge_index)
            sim = F.cosine_similarity(z[current_node].unsqueeze(0), z)
            
            topk = torch.topk(sim, self.top_k + 1).indices.tolist()
            if current_node in topk:
                topk.remove(current_node)
            return topk[:self.top_k]

    def prefetch(self, current_node):
        predicted_nodes = self.predict_next(current_node)
        print(f"\n🔮 Prefetching for Node {current_node}: Predicted Next Nodes -> {predicted_nodes}")
        
        for node_id in predicted_nodes:
            self.cache.prefetch(node_id)
        self.cache.show()

if __name__ == "__main__":
    cache = LocalCache()
    manager = PrefetchManager(model, data, cache, top_k=5)

    for node_id in [5, 20, 35]:
        print(f"\n📂 User accessed Node {node_id}")
        manager.prefetch(node_id)
