"""生成模拟网络攻击场景数据，用于演示上传推理功能

核心策略：从真实训练数据中采样子图，确保特征分布与训练数据一致，
让模型能给出有区分度的推理结果。

用法：
    python generate_demo_data.py
    python generate_demo_data.py --num_nodes 150 --output_dir demo_data
"""
import sys
import torch
import numpy as np
from pathlib import Path
from collections import deque


def sample_subgraph_from_real_data(base_dir, num_target=100, seed=42):
    """从真实训练数据中采样一个子图

    策略：
    1. 先用训练好的模型推理，找到高概率恶意节点和低概率正常节点
    2. 以部分恶意节点为种子，BFS 扩展出一个子图
    3. 补充一些正常节点，凑够目标数量
    """
    np.random.seed(seed)
    base = Path(base_dir)

    # 加载完整图数据
    features = torch.load(base / 'features.pt', weights_only=False)
    edge_index = torch.load(base / 'edge_index.pt', weights_only=False)
    edge_type = torch.load(base / 'edge_type.pt', weights_only=False)
    num_nodes = features.size(0)
    feat_dim = features.size(1)

    print(f"原始图: {num_nodes} 节点, {edge_index.size(1)} 边, {feat_dim} 维特征")

    # 尝试加载模型做推理，获取节点概率
    model_path = base / 'results' / 'ge-mrgcn' / 'rgcn_group_model.pt'
    probs = None
    if model_path.exists():
        try:
            from inference import predict as do_predict
            from train_mrgnn import MR_GNN
            checkpoint = torch.load(model_path, map_location='cpu', weights_only=False)
            model = MR_GNN(
                checkpoint['in_dim'], checkpoint['hidden_dim'],
                checkpoint['out_dim'], checkpoint['num_relations'],
                model_type=checkpoint['model_type']
            )
            model.load_state_dict(checkpoint['model_state_dict'])
            model.eval()

            edge_weight = None
            if (base / 'edge_weight.pt').exists():
                edge_weight = torch.load(base / 'edge_weight.pt', weights_only=False)
                ew_min, ew_max = edge_weight.min(), edge_weight.max()
                if ew_max - ew_min > 1e-8:
                    edge_weight = (edge_weight - ew_min) / (ew_max - ew_min) + 0.1

            probs = do_predict(model, features, edge_index, edge_type, edge_weight)
            print(f"模型推理完成，概率范围: [{probs.min():.3f}, {probs.max():.3f}]")
        except Exception as e:
            print(f"模型推理失败({e})，使用随机采样")

    # --- 选择种子节点 ---
    if probs is not None:
        # 选一些高概率（恶意）和低概率（正常）节点
        sorted_idx = torch.argsort(probs, descending=True)
        n_mal = max(5, int(num_target * 0.15))   # ~15% 恶意
        n_normal = num_target - n_mal

        # 恶意种子：概率最高的一批
        mal_seeds = sorted_idx[:n_mal * 3].tolist()  # 多取一些备选
        np.random.shuffle(mal_seeds)
        mal_seeds = mal_seeds[:n_mal]

        # 正常种子：概率最低的一批
        normal_pool = sorted_idx[-num_nodes // 3:].tolist()
        np.random.shuffle(normal_pool)
    else:
        # 没有模型，随机选
        all_ids = list(range(num_nodes))
        np.random.shuffle(all_ids)
        n_mal = max(5, int(num_target * 0.15))
        n_normal = num_target - n_mal
        mal_seeds = all_ids[:n_mal]
        normal_pool = all_ids[n_mal:]

    # --- 构建邻接表 ---
    src_arr = edge_index[0].tolist()
    dst_arr = edge_index[1].tolist()
    typ_arr = edge_type.tolist()
    adj = {}
    for s, d in zip(src_arr, dst_arr):
        adj.setdefault(s, []).append(d)
        adj.setdefault(d, []).append(s)

    # --- BFS 从恶意种子扩展，收集附近节点 ---
    selected = set()
    for seed in mal_seeds:
        if len(selected) >= num_target:
            break
        queue = deque([(seed, 0)])
        while queue and len(selected) < num_target:
            node, depth = queue.popleft()
            if node in selected:
                continue
            selected.add(node)
            if depth < 1:  # 1-hop 邻居
                neighbors = adj.get(node, [])
                np.random.shuffle(neighbors)
                for nb in neighbors[:5]:  # 限制每个节点最多扩展 5 个邻居
                    if nb not in selected:
                        queue.append((nb, depth + 1))

    # 如果不够，从正常池补充
    for nid in normal_pool:
        if len(selected) >= num_target:
            break
        if nid not in selected:
            selected.add(nid)

    selected = sorted(selected)[:num_target]
    print(f"采样了 {len(selected)} 个节点")

    # --- 构建子图 ---
    # 节点重映射
    old2new = {old: new for new, old in enumerate(selected)}
    new_set = set(selected)

    # 提取子图特征
    sub_features = features[selected]

    # 提取子图边
    new_src, new_dst, new_type = [], [], []
    for s, d, t in zip(src_arr, dst_arr, typ_arr):
        if s in new_set and d in new_set:
            new_src.append(old2new[s])
            new_dst.append(old2new[d])
            new_type.append(t)

    sub_edge_index = torch.tensor([new_src, new_dst], dtype=torch.long) if new_src else torch.zeros(2, 0, dtype=torch.long)
    sub_edge_type = torch.tensor(new_type, dtype=torch.long) if new_type else torch.zeros(0, dtype=torch.long)

    # 记录哪些是恶意节点（新编号）
    mal_new_ids = []
    if probs is not None:
        for new_id, old_id in enumerate(selected):
            if probs[old_id].item() > 0.6:
                mal_new_ids.append(new_id)
    else:
        mal_new_ids = list(range(n_mal))

    # IP 标签
    ip_map = {}
    for new_id, old_id in enumerate(selected):
        if new_id in mal_new_ids:
            ip_map[new_id] = f"10.0.{new_id // 256}.{new_id % 256 + 1}"
        else:
            ip_map[new_id] = f"192.168.{new_id // 256}.{new_id % 256 + 1}"

    return sub_features, sub_edge_index, sub_edge_type, mal_new_ids, ip_map, selected


def main():
    args = sys.argv
    num_nodes = 100
    base_dir = Path(__file__).parent
    output_dir = base_dir / 'demo_data'

    if '--num_nodes' in args:
        num_nodes = int(args[args.index('--num_nodes') + 1])
    if '--output_dir' in args:
        output_dir = Path(args[args.index('--output_dir') + 1])

    output_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 50)
    print("从真实数据采样演示场景")
    print("=" * 50)

    features, edge_index, edge_type, mal_ids, ip_map, orig_ids = \
        sample_subgraph_from_real_data(base_dir, num_target=num_nodes)

    n_nodes = features.size(0)
    n_edges = edge_index.size(1) if edge_index.numel() > 0 else 0
    print(f"\n子图统计:")
    print(f"  节点数: {n_nodes} (预期恶意: {len(mal_ids)})")
    print(f"  边数:   {n_edges}")
    print(f"  特征维度: {features.size(1)}")
    if n_edges > 0:
        svc_names = ['HTTP/HTTPS', 'SSH/远程', 'DNS', '邮件', '高端口扫描', '其他']
        print(f"  边类型分布:")
        for t in range(6):
            cnt = (edge_type == t).sum().item()
            if cnt > 0:
                name = svc_names[t] if t < len(svc_names) else f'类型{t}'
                print(f"    {name}: {cnt} 条")

    # 保存
    torch.save(features, output_dir / 'features.pt')
    torch.save(edge_index, output_dir / 'edge_index.pt')
    torch.save(edge_type, output_dir / 'edge_type.pt')

    info = f"总节点: {n_nodes}\n"
    info += f"预期恶意节点数: {len(mal_ids)}\n"
    info += f"预期恶意节点ID: {mal_ids}\n"
    info += f"原始图中对应的节点: {[orig_ids[i] for i in mal_ids]}\n"
    info += f"攻击者IP:\n" + "\n".join(f"  {i}: {ip_map[i]}" for i in mal_ids)
    (output_dir / 'scenario_info.txt').write_text(info, encoding='utf-8')

    print(f"\n已保存到: {output_dir}")
    print(f"\n=== 演示步骤 ===")
    print(f"1. 启动服务: python api_server.py")
    print(f"2. 打开浏览器: http://127.0.0.1:8000")
    print(f"3. 点击「数据推理」→ 上传 demo_data/ 下的 3 个 .pt 文件")
    print(f"4. 查看推理结果，预期检出 ~{len(mal_ids)} 个恶意节点")


if __name__ == '__main__':
    main()
