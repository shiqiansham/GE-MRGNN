"""GE-MRGNN 离线推理脚本

功能：
1. 加载训练好的模型，对图数据做推理
2. 输出恶意节点列表 (CSV)
3. 提取团伙子图 (JSON)：以高置信恶意节点为种子，BFS 扩展连通子图
4. 支持阈值选择策略

用法：
    python inference.py --model_path results/rgcn_group/rgcn_group_model.pt --output_dir inference_output
    python inference.py --model_path results/rgcn_group/rgcn_group_model.pt --threshold 0.7
"""
import sys
import json
import csv
from pathlib import Path
from collections import deque

import torch
import numpy as np


def load_model_and_data(model_path, dataset_dir):
    """加载模型与图数据"""
    # 延迟导入，避免循环依赖
    from train_mrgnn import MR_GNN, HANModel, DGIClassifier

    checkpoint = torch.load(model_path, map_location='cpu')
    model_type = checkpoint['model_type']
    hidden_dim = checkpoint['hidden_dim']
    num_relations = checkpoint['num_relations']
    in_dim = checkpoint['in_dim']
    out_dim = checkpoint['out_dim']

    if model_type == 'han':
        model = HANModel(in_dim, hidden_dim, out_dim, num_relations)
    elif model_type == 'dgi':
        model = DGIClassifier(in_dim, hidden_dim, out_dim)
    else:
        model = MR_GNN(in_dim, hidden_dim, out_dim, num_relations, model_type=model_type)

    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()

    # 加载图数据
    base = Path(dataset_dir)
    features = torch.load(base / 'features.pt')
    edge_index = torch.load(base / 'edge_index.pt')
    edge_type = torch.load(base / 'edge_type.pt')
    edge_weight = None
    if (base / 'edge_weight.pt').exists():
        edge_weight = torch.load(base / 'edge_weight.pt')
        ew_min, ew_max = edge_weight.min(), edge_weight.max()
        if ew_max - ew_min > 1e-8:
            edge_weight = (edge_weight - ew_min) / (ew_max - ew_min) + 0.1
        else:
            edge_weight = torch.ones_like(edge_weight)

    # 如果是 GCN，忽略关系类型
    if model_type == 'gcn':
        edge_type = torch.zeros_like(edge_type)

    # IP 映射（如果有）
    ip_map = {}
    ip_index_file = base / 'ip_index.txt'
    if ip_index_file.exists():
        for line in ip_index_file.read_text(encoding='utf-8').splitlines():
            parts = line.split(',', 1)
            if len(parts) == 2:
                ip_map[int(parts[0])] = parts[1]

    return model, features, edge_index, edge_type, edge_weight, ip_map


def predict(model, features, edge_index, edge_type, edge_weight=None):
    """推理，返回每个节点的恶意概率"""
    with torch.no_grad():
        logits = model(features, edge_index, edge_type, edge_weight)
        probs = torch.sigmoid(logits.squeeze(-1))
    return probs


def extract_gang_subgraphs(malicious_nodes, edge_index, max_hops=1, max_gang_nodes=200):
    """以恶意节点为种子，BFS 扩展团伙子图（高性能版）

    优化策略:
    - 用 Tensor 操作构建邻接表，避免 Python 逐边循环
    - 限制单个子图最大节点数，防止稠密图爆炸
    - 用 set 做边过滤，O(1) 查询

    Args:
        malicious_nodes: 恶意节点列表
        edge_index: [2, E] 边张量
        max_hops: BFS 扩展跳数（稠密图建议用 1）
        max_gang_nodes: 单个子图最大节点数

    Returns:
        list of dict: 每个连通子图
    """
    mal_set = set(malicious_nodes)

    # 构建邻接表（向量化）
    src_arr = edge_index[0].tolist()
    dst_arr = edge_index[1].tolist()
    adj = {}
    edge_set = set()
    for s, d in zip(src_arr, dst_arr):
        adj.setdefault(s, []).append(d)
        adj.setdefault(d, []).append(s)
        edge_set.add((s, d))

    visited = set()
    subgraphs = []

    for seed in malicious_nodes:
        if seed in visited:
            continue
        # BFS（带节点数上限）
        queue = deque([(seed, 0)])
        sg_nodes = set()
        while queue:
            node, depth = queue.popleft()
            if node in sg_nodes:
                continue
            sg_nodes.add(node)
            if len(sg_nodes) >= max_gang_nodes:
                break
            if depth < max_hops:
                for nb in adj.get(node, []):
                    if nb not in sg_nodes:
                        queue.append((nb, depth + 1))
        visited.update(sg_nodes)

        # 提取子图边（O(节点度) 而非 O(全部边)）
        sg_edges = []
        for n in sg_nodes:
            for nb in adj.get(n, []):
                if nb in sg_nodes and (n, nb) in edge_set:
                    sg_edges.append([n, nb])

        if len(sg_nodes) >= 2:
            subgraphs.append({
                "nodes": sorted(sg_nodes),
                "edges": sg_edges,
                "num_malicious": len(sg_nodes & mal_set),
                "size": len(sg_nodes)
            })

    subgraphs.sort(key=lambda g: g['num_malicious'], reverse=True)
    return subgraphs


def main():
    args = sys.argv

    def get_arg(name, default=None, cast=str):
        if name in args:
            try:
                return cast(args[args.index(name) + 1])
            except Exception:
                return default
        return default

    model_path = get_arg('--model_path', 'results/ge-mrgcn/rgcn_group_model.pt')
    dataset_dir = get_arg('--dataset', str(Path(__file__).parent))
    output_dir = Path(get_arg('--output_dir', 'inference_output'))
    threshold = get_arg('--threshold', 0.5, float)
    max_hops = get_arg('--max_hops', 2, int)
    top_k = get_arg('--top_k', None, int)

    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"加载模型: {model_path}")
    model, features, edge_index, edge_type, edge_weight, ip_map = load_model_and_data(model_path, dataset_dir)
    num_nodes = features.size(0)

    print(f"推理中... (节点数={num_nodes}, 边数={edge_index.size(1)})")
    probs = predict(model, features, edge_index, edge_type, edge_weight)

    # 恶意节点列表
    if top_k is not None:
        top_indices = torch.argsort(probs, descending=True)[:top_k]
        malicious_mask = torch.zeros(num_nodes, dtype=torch.bool)
        malicious_mask[top_indices] = True
    else:
        malicious_mask = probs > threshold

    malicious_nodes = malicious_mask.nonzero(as_tuple=True)[0].tolist()
    print(f"检测到恶意节点: {len(malicious_nodes)} (阈值={threshold})")

    # 输出恶意节点 CSV
    nodes_csv = output_dir / 'malicious_nodes.csv'
    with open(nodes_csv, 'w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerow(['node_id', 'ip', 'probability', 'is_malicious'])
        order = torch.argsort(probs, descending=True)
        for idx in order.tolist():
            ip = ip_map.get(idx, f"node_{idx}")
            writer.writerow([idx, ip, f"{probs[idx].item():.6f}", int(malicious_mask[idx].item())])
    print(f"节点预测结果已保存: {nodes_csv}")

    # 提取团伙子图
    print(f"提取团伙子图 (BFS max_hops={max_hops})...")
    subgraphs = extract_gang_subgraphs(malicious_nodes, edge_index, max_hops=max_hops)
    print(f"发现 {len(subgraphs)} 个团伙子图")

    # 输出子图 JSON（节点带 IP 标注）
    for i, sg in enumerate(subgraphs):
        sg['node_labels'] = {str(n): ip_map.get(n, f"node_{n}") for n in sg['nodes']}
        sg['node_probs'] = {str(n): round(probs[n].item(), 6) for n in sg['nodes']}

    gangs_json = output_dir / 'gang_subgraphs.json'
    with open(gangs_json, 'w', encoding='utf-8') as f:
        json.dump(subgraphs, f, indent=2, ensure_ascii=False)
    print(f"团伙子图已保存: {gangs_json}")

    # 输出简要统计
    stats = {
        "total_nodes": num_nodes,
        "total_edges": edge_index.size(1),
        "threshold": threshold,
        "num_malicious": len(malicious_nodes),
        "num_gangs": len(subgraphs),
        "gang_sizes": [sg['size'] for sg in subgraphs[:20]],
        "top10_malicious": [
            {"node_id": int(order[i]), "ip": ip_map.get(int(order[i]), ""), "prob": round(probs[order[i]].item(), 6)}
            for i in range(min(10, num_nodes))
        ]
    }
    stats_json = output_dir / 'inference_stats.json'
    with open(stats_json, 'w', encoding='utf-8') as f:
        json.dump(stats, f, indent=2, ensure_ascii=False)
    print(f"统计信息已保存: {stats_json}")

    print("\n推理完成!")


if __name__ == '__main__':
    main()
