import csv
import sys
from pathlib import Path
import torch

def read_nodes(path):
    ids = []
    feats = []
    with open(path, newline='', encoding='utf-8') as f:
        r = csv.reader(f)
        for row in r:
            if not row:
                continue
            ids.append(row[0])
            vals = [float(x) for x in row[1:]]
            feats.append(vals)
    id2idx = {nid: i for i, nid in enumerate(ids)}
    features = torch.tensor(feats, dtype=torch.float32)
    return id2idx, features

def read_edges(path, id2idx):
    src = []
    dst = []
    etype = []
    eweight = []
    with open(path, newline='', encoding='utf-8') as f:
        r = csv.DictReader(f)
        if r.fieldnames is None:
            f.seek(0)
            r = csv.reader(f)
            for row in r:
                if len(row) < 2:
                    continue
                s = id2idx.get(row[0])
                d = id2idx.get(row[1])
                if s is None or d is None:
                    continue
                src.append(s)
                dst.append(d)
                etype.append(0)
                eweight.append(1.0)
        else:
            for row in r:
                s = id2idx.get(row.get('src') or row.get('source') or row.get('u') or row.get('from'))
                d = id2idx.get(row.get('dst') or row.get('target') or row.get('v') or row.get('to'))
                if s is None or d is None:
                    continue
                src.append(s)
                dst.append(d)
                t = row.get('type')
                w = row.get('weight')
                etype.append(int(t) if t is not None and t != '' else 0)
                eweight.append(float(w) if w is not None and w != '' else 1.0)
    edge_index = torch.tensor([src, dst], dtype=torch.long)
    edge_type = torch.tensor(etype, dtype=torch.long)
    edge_weight = torch.tensor(eweight, dtype=torch.float32)
    return edge_index, edge_type, edge_weight

def read_labels(path, id2idx, num_nodes):
    labels = torch.full((num_nodes,), -1, dtype=torch.long)
    if not Path(path).exists():
        return labels
    with open(path, newline='', encoding='utf-8') as f:
        r = csv.reader(f)
        for row in r:
            if len(row) < 2:
                continue
            nid = row[0]
            lab = int(row[1])
            idx = id2idx.get(nid)
            if idx is not None:
                labels[idx] = lab
    return labels

def main():
    if '--input_dir' not in sys.argv:
        return
    i = sys.argv.index('--input_dir')
    in_dir = Path(sys.argv[i + 1])
    out_dir = in_dir if '--output_dir' not in sys.argv else Path(sys.argv[sys.argv.index('--output_dir') + 1])
    id2idx, features = read_nodes(in_dir / 'nodes.csv')
    edge_index, edge_type, edge_weight = read_edges(in_dir / 'edges.csv', id2idx)
    labels_bot = read_labels(in_dir / 'labels_bot.csv', id2idx, features.size(0))
    labels_stance = read_labels(in_dir / 'labels_stance.csv', id2idx, features.size(0))
    torch.save(features, out_dir / 'features.pt')
    torch.save(edge_index, out_dir / 'edge_index.pt')
    torch.save(edge_type, out_dir / 'edge_type.pt')
    torch.save(edge_weight, out_dir / 'edge_weight.pt')
    if (labels_bot >= 0).any():
        torch.save(labels_bot, out_dir / 'labels_bot.pt')
    if (labels_stance >= 0).any():
        torch.save(labels_stance, out_dir / 'labels_stance.pt')

if __name__ == '__main__':
    main()
