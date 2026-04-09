import torch
import torch.nn.functional as F
import os
import argparse
from model_dmon import DMON_DPR_Model

def train_dmon_dpr(data_dir, model_dir, K_val, W_dist_val, W_var_val, run_id):
    os.makedirs(model_dir, exist_ok=True)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Încărcare dinamică
    data = torch.load(f'{data_dir}/hetero_graph.pt', weights_only=False).to(device)
    x = data['patent'].x 
    edge_index = data['patent', 'cites', 'patent'].edge_index
    n_nodes = x.size(0)
    n_edges = edge_index.size(1)

    K = K_val
    W_dist = W_dist_val
    W_var = W_var_val   
    W_entropy = 0.1 
    epsilon = 1.0   

    model = DMON_DPR_Model(x.size(1), 512, K).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

    adj = torch.sparse_coo_tensor(edge_index, torch.ones(n_edges, device=device), (n_nodes, n_nodes))
    degrees = torch.sparse.sum(adj, dim=1).to_dense().view(-1, 1)

    model.train()
    for epoch in range(1, 201):
        optimizer.zero_grad()
        x_latent, c = model(x, edge_index)

        null_model = (c.t() @ degrees) @ (degrees.t() @ c) / (2 * n_edges)
        l_mod = -(torch.trace(c.t() @ adj @ c) - torch.trace(null_model)) / (2 * n_edges)
        
        l_collapse = (torch.sqrt(torch.tensor(K).float()) / n_nodes) * torch.norm(c.sum(dim=0), p='fro') - 1

        mu = (c.t() @ x) / (c.sum(dim=0).view(-1, 1) + 1e-15) 
        dist_matrix = torch.cdist(mu, mu, p=2)**2
        l_dist = (1.0 / (K * (K - 1))) * torch.sum(F.relu(epsilon - dist_matrix))

        l_var = -torch.mean(torch.var(c, dim=0))
        l_entropy = -torch.mean(torch.sum(c * torch.log(c + 1e-15), dim=1))

        loss = l_mod + l_collapse + W_dist * l_dist + W_var * l_var + W_entropy * l_entropy
        
        loss.backward()
        optimizer.step()
        
        if epoch % 20 == 0:
            print(f"Epoch {epoch:03d} | Modularity: {-l_mod.item():.4f} | Dist: {l_dist.item():.4f}")

    # Salvare dinamică
    torch.save(model.state_dict(), f'{model_dir}/dmon_dpr_model_{run_id}.pth')
    print("Antrenare finalizată cu algoritmul DMON-DPR.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir', type=str, required=True)
    parser.add_argument('--model_dir', type=str, required=True)
    parser.add_argument('--K', type=int, required=True)
    parser.add_argument('--W_dist', type=float, required=True)
    parser.add_argument('--W_var', type=float, required=True)
    parser.add_argument('--run_id', type=str, required=True)
    args = parser.parse_args()
    train_dmon_dpr(args.data_dir, args.model_dir, args.K, args.W_dist, args.W_var, args.run_id)