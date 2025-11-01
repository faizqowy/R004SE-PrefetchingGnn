import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.data import Data
from torch_geometric.nn import SAGEConv, GCNConv, GATConv
import random

torch.serialization.add_safe_globals([Data])

# GNN Model for Embedding Learning

class GNNPrefetch(nn.Module):
    def __init__(self, in_dim, hidden_dim, emb_dim, num_layers=2, dropout=0.3, model_type="sage", heads=4):
        super(GNNPrefetch, self).__init__()
        self.convs = nn.ModuleList()
        self.model_type = model_type.lower()
        self.dropout = dropout

        if self.model_type == "sage":
            conv_class = SAGEConv
        elif self.model_type == "gcn":
            conv_class = GCNConv
        elif self.model_type == "gat":
            conv_class = lambda in_c, out_c: GATConv(in_c, out_c // heads, heads=heads)
        else:
            raise ValueError(f"Unsupported model_type: {model_type}")

        self.convs.append(conv_class(in_dim, hidden_dim))
        for _ in range(num_layers - 2):
            self.convs.append(conv_class(hidden_dim, hidden_dim))
        self.convs.append(conv_class(hidden_dim, emb_dim))

    def forward(self, x, edge_index):
        for conv in self.convs[:-1]:
            x = conv(x, edge_index)
            x = F.relu(x)
            x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.convs[-1](x, edge_index)
        return x

    def get_embedding(self, x, edge_index):
        # Return node embeddings before the final output layer
        for conv in self.convs[:-1]:
            x = conv(x, edge_index)
            x = F.relu(x)
        return x

def split_dataset(data, train_ratio=0.7, val_ratio=0.15):
    print("Unique src nodes:", len(torch.unique(data.src)))
    print("Unique target nodes:", len(torch.unique(data.y)))
    print("Total nodes:", data.x.size(0))

    num_samples = data.src.size(0)
    indices = list(range(num_samples))
    random.shuffle(indices)

    train_end = int(train_ratio * num_samples)
    val_end = int((train_ratio + val_ratio) * num_samples)

    train_idx = indices[:train_end]
    val_idx = indices[train_end:val_end]
    test_idx = indices[val_end:]

    return train_idx, val_idx, test_idx

# Link Prediction Training Step

def train_linkpred(model, data, train_idx, optimizer):
    model.train()
    optimizer.zero_grad()

    z = model(data.x, data.edge_index)

    pos_src = data.src[train_idx]
    pos_tgt = data.y[train_idx]

    # Sample random negatives
    neg_tgt = torch.randint(0, z.size(0), pos_tgt.size(), device=z.device)

    # Compute similarity (dot product)
    pos_score = (z[pos_src] * z[pos_tgt]).sum(dim=1)
    neg_score = (z[pos_src] * z[neg_tgt]).sum(dim=1)

    # Contrastive loss: push positive scores higher than negatives
    loss = -torch.log(torch.sigmoid(pos_score - neg_score)).mean()

    loss.backward()
    optimizer.step()

    return loss.item()

# Evaluation: Top-K Hit Rate

@torch.no_grad()
def evaluate_linkpred(model, data, idx, top_k=5):
    model.eval()
    z = model(data.x, data.edge_index)

    correct = 0
    total = len(idx)

    for s, t in zip(data.src[idx], data.y[idx]):
        scores = (z[s] @ z.T)  # similarity to all nodes
        top_pred = torch.topk(scores, top_k).indices
        if t in top_pred:
            correct += 1

    hit_rate = correct / total
    return hit_rate

# Training Loop

if __name__ == "__main__":
    data: Data = torch.load(
        "E:/Research/R004SE-PrefetchingGnn/user-session-sim/gnn_prefetch_dataset.pt",
        weights_only=False
    )

    in_dim = data.x.size(1)
    hidden_dim = 128
    emb_dim = 64

    model = GNNPrefetch(in_dim, hidden_dim, emb_dim, num_layers=3, model_type="sage")
    optimizer = torch.optim.Adam(model.parameters(), lr=0.005, weight_decay=1e-4)

    train_idx, val_idx, test_idx = split_dataset(data)

    print(f"Train: {len(train_idx)}, Val: {len(val_idx)}, Test: {len(test_idx)}")

    epochs = 1000
    for epoch in range(1, epochs + 1):
        loss = train_linkpred(model, data, train_idx, optimizer)
        val_hit = evaluate_linkpred(model, data, val_idx, top_k=5)
        print(f"Epoch {epoch:03d} | Loss: {loss:.4f} | Val Top-5 HitRate: {val_hit:.4f}")

    test_hit = evaluate_linkpred(model, data, test_idx, top_k=5)
    print(f"✅ Test Top-5 HitRate: {test_hit:.4f}")

    torch.save(model.state_dict(), "gnn_prefetch_linkpred.pt")
    print("💾 Model saved as gnn_prefetch_linkpred.pt")
