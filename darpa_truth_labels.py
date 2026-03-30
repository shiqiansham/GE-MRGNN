import sys
from pathlib import Path
import csv
import torch

def main():
    if '--dataset' not in sys.argv:
        return
    base = Path(sys.argv[sys.argv.index('--dataset') + 1])
    if '--make_template' in sys.argv:
        ip_map = {}
        ip_index_file = base / 'ip_index.txt'
        if ip_index_file.exists():
            pairs = []
            for line in ip_index_file.read_text(encoding='utf-8').splitlines():
                idx_str, ip = line.split(',', 1)
                pairs.append((int(idx_str), ip))
            pairs.sort(key=lambda x: x[0])
            out = base / 'truth_template.csv'
            with open(out, 'w', encoding='utf-8', newline='') as f:
                w = csv.writer(f)
                for _, ip in pairs:
                    w.writerow([ip, ""])
        return
    if '--truth' not in sys.argv:
        return
    truth = Path(sys.argv[sys.argv.index('--truth') + 1])
    ip_map = {}
    ip_index_file = base / 'ip_index.txt'
    if ip_index_file.exists():
        for line in ip_index_file.read_text(encoding='utf-8').splitlines():
            idx_str, ip = line.split(',', 1)
            ip_map[ip] = int(idx_str)
    features = torch.load(base / 'features.pt')
    labels = torch.full((features.size(0),), -1, dtype=torch.long)
    with open(truth, newline='', encoding='utf-8') as f:
        r = csv.reader(f)
        for row in r:
            if len(row) < 2 or (row[1] is None) or (str(row[1]).strip() == ''):
                continue
            ip = row[0]
            lab = int(row[1])
            idx = ip_map.get(ip)
            if idx is not None:
                labels[idx] = lab
    torch.save(labels, base / 'labels_bot.pt')

if __name__ == '__main__':
    main()
