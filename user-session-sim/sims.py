import random
import json
import networkx as nx
import torch
from torch_geometric.data import Data


def load_graph(graph_file):
    return nx.read_gexf(graph_file)


# Return nodes that have no predecessors
def get_root_nodes(G):
    return [n for n in G.nodes if len(list(G.predecessors(n))) == 0]


def get_directory_nodes(G):
    return [
        n for n, data in G.nodes(data=True)
        if str(data.get("is_directory", "false")).lower() == "true"
    ]


def simulate_user_sessions(
    G,
    num_projects=10,
    users_per_project=10,
    sessions_per_user=10,
    max_steps=30,
    down_prob=0.8,
    back_prob=0.1,
    jump_prob=0.05,
    seed=42,
):
    random.seed(seed)
    all_sessions = {}

    root_nodes = get_root_nodes(G)
    if not root_nodes:
        print("⚠️ No explicit roots found; using all directories as roots.")
        root_nodes = get_directory_nodes(G)

    selected_projects = random.sample(root_nodes, min(num_projects, len(root_nodes)))
    user_counter = 1

    for project_root in selected_projects:
        project_name = G.nodes[project_root].get("node_name", str(project_root))
        project_files = [n for n in nx.descendants(G, project_root)]

        # Skip empty projects
        if not project_files:
            continue

        for _ in range(users_per_project):
            user_id = f"user_{user_counter:03d}_proj_{project_name}"
            user_counter += 1
            sessions = []

            for _ in range(sessions_per_user):
                session = []
                current = project_root

                # Randomize depth per session
                steps = random.randint(int(max_steps * 0.4), max_steps)
                for _ in range(steps):
                    session.append(current)
                    neighbors = list(G.successors(current))

                    if not neighbors:
                        break

                    # Weighted choice — prefer nodes with fewer files (more focused exploration)
                    weights = [1.0 / (G.degree(v) + 1e-3) for v in neighbors]

                    r = random.random()
                    if r < down_prob:
                        current = random.choices(neighbors, weights=weights, k=1)[0]
                    elif r < down_prob + back_prob and len(session) > 1:
                        current = session[-2]
                    elif r < down_prob + back_prob + jump_prob:
                        current = random.choice(selected_projects)
                    else:
                        break

                    # stop early at a file
                    if str(G.nodes[current].get("is_directory", "false")).lower() != "true":
                        session.append(current)
                        break

                if len(session) > 1:
                    sessions.append(session)

            all_sessions[user_id] = sessions

    return all_sessions


def save_sessions(all_users_sessions, G, out_file="user_sessions.json"):
    readable_data = {}

    for user, sessions in all_users_sessions.items():
        readable_data[user] = []
        for session in sessions:
            readable_data[user].append([
                G.nodes[n].get("node_name", str(n)) for n in session
            ])

    with open(out_file, "w", encoding="utf-8") as f:
        json.dump(readable_data, f, indent=2, ensure_ascii=False)

    print(f"✅ Saved sessions for {len(all_users_sessions)} users to {out_file}")


def build_gnn_dataset(G, sessions, feature_type="onehot"):
    node_mapping = {n: i for i, n in enumerate(G.nodes())}
    num_nodes = len(node_mapping)

    if feature_type == "onehot":
        x = torch.eye(num_nodes)
    elif feature_type == "degree":
        degs = [G.degree(n) for n in G.nodes()]
        x = torch.tensor(degs, dtype=torch.float).unsqueeze(1)
    else:
        raise ValueError("Unknown feature_type")

    edges = [(node_mapping[u], node_mapping[v]) for u, v in G.edges()]
    edge_index = torch.tensor(edges, dtype=torch.long).t().contiguous()

    src_nodes = []
    tgt_nodes = []
    for user, sess_list in sessions.items():
        for sess in sess_list:
            for i in range(len(sess) - 1):
                src_nodes.append(node_mapping[sess[i]])
                tgt_nodes.append(node_mapping[sess[i + 1]])

    y = torch.tensor(tgt_nodes, dtype=torch.long)
    src = torch.tensor(src_nodes, dtype=torch.long)

    data = Data(x=x, edge_index=edge_index, src=src, y=y)
    return data


if __name__ == "__main__":
    G = load_graph("E:/Research/R004SE-PrefetchingGnn/route-mapper/graph.gexf")

    all_users_sessions = simulate_user_sessions(
        G,
        num_projects=10,
        users_per_project=15,
        sessions_per_user=20,
        max_steps=50,
        down_prob=0.78,
        back_prob=0.15,
        jump_prob=0.07,
        seed=123,
    )

    save_sessions(all_users_sessions, G)

    gnn_data = build_gnn_dataset(G, all_users_sessions, feature_type="onehot")
    torch.save(gnn_data, "gnn_prefetch_dataset.pt")
    print("✅ Saved GNN dataset to gnn_prefetch_dataset.pt")
