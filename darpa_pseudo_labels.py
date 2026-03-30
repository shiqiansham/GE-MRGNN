import sys
from pathlib import Path
import torch

def minmax(x):
    mn = x.min().item()
    mx = x.max().item()
    if mx - mn < 1e-12:
        return torch.zeros_like(x)
    return (x - mn) / (mx - mn)

def main():
    if '--dataset' not in sys.argv:
        return
    base = Path(sys.argv[sys.argv.index('--dataset') + 1])
    top_ratio = 0.1
    if '--top_ratio' in sys.argv:
        try:
            top_ratio = float(sys.argv[sys.argv.index('--top_ratio') + 1])
        except:
            top_ratio = 0.1
    features = torch.load(base / 'features.pt')
    edge_index = torch.load(base / 'edge_index.pt')
    num_nodes = features.size(0)
    deg_out = torch.zeros((num_nodes,), dtype=torch.float32)
    deg_out.index_add_(0, edge_index[0], torch.ones(edge_index.size(1)))
    deg_in = torch.zeros((num_nodes,), dtype=torch.float32)
    deg_in.index_add_(0, edge_index[1], torch.ones(edge_index.size(1)))
    in_bytes = features[:, 0]
    out_bytes = features[:, 1]
    in_pkts = features[:, 2]
    out_pkts = features[:, 3]
    tcp_in = features[:, 4]
    tcp_out = features[:, 5]
    udp_in = features[:, 6]
    udp_out = features[:, 7]
    score = (
        0.35 * minmax(out_bytes) +
        0.25 * minmax(out_pkts) +
        0.20 * minmax(deg_out) +
        0.10 * minmax(tcp_out) +
        0.10 * minmax(udp_out)
    )
    k = max(1, int(num_nodes * top_ratio))
    order = torch.argsort(score, descending=True)
    labels = torch.zeros((num_nodes,), dtype=torch.long)
    labels[order[:k]] = 1
    torch.save(labels, base / 'labels_bot.pt')
    (base / 'pseudo_labels_info.txt').write_text(
        f'num_nodes={num_nodes}\n'
        f'top_ratio={top_ratio}\n'
        f'top_k={k}\n'
        f'score_min={score.min().item():.6f}\n'
        f'score_max={score.max().item():.6f}\n',
        encoding='utf-8'
    )

if __name__ == '__main__':
    main()
