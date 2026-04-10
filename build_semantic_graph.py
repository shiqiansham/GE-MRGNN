"""多关系安全攻击图谱语义构建

基于 DARPA pcap 数据，构建更丰富的多关系异质图。

扩展内容（相比 darpa_pcap_to_pt.py）：
1. 边类型从 2 种 (TCP/UDP) 扩展到 6 种服务语义类型
   - 0: HTTP/HTTPS (端口 80, 443, 8080, 8443)
   - 1: SSH/远程 (端口 22, 23, 3389)
   - 2: DNS (端口 53)
   - 3: 邮件 (端口 25, 110, 143, 465, 587, 993, 995)
   - 4: 高端口扫描 (src/dst 端口均 > 1024 且连接数少)
   - 5: 其他服务

2. 节点特征从 8 维扩展到 20 维
   原始: [in_bytes, out_bytes, in_pkts, out_pkts, tcp_in, tcp_out, udp_in, udp_out]
   新增: [unique_dst_ports, unique_src_ports, unique_peers,
          http_ratio, ssh_ratio, dns_ratio, mail_ratio, scan_ratio,
          avg_pkt_size, max_burst_rate, in_out_byte_ratio, port_entropy]

3. 边权重改为连接次数（反映通信强度）

用法：
    python build_semantic_graph.py --input_dir <pcap目录> --output_dir <输出目录>
    python build_semantic_graph.py --upgrade   # 基于现有 .pt 文件增强
"""
import sys
import math
import struct
import gzip
from pathlib import Path
from collections import defaultdict

import torch
import numpy as np


# ---------------------------------------------------------------------------
#  服务语义映射
# ---------------------------------------------------------------------------

SERVICE_PORTS = {
    # HTTP/HTTPS
    80: 0, 443: 0, 8080: 0, 8443: 0, 8000: 0, 3000: 0,
    # SSH / Telnet / RDP
    22: 1, 23: 1, 3389: 1,
    # DNS
    53: 2,
    # 邮件 (SMTP, POP3, IMAP 及其 SSL 变体)
    25: 3, 110: 3, 143: 3, 465: 3, 587: 3, 993: 3, 995: 3,
}


def classify_edge(proto, sport, dport):
    """根据协议和端口将边分类为语义类型

    返回边类型编号 (0-5)
    """
    # 优先匹配目的端口（服务端），再匹配源端口
    if dport in SERVICE_PORTS:
        return SERVICE_PORTS[dport]
    if sport in SERVICE_PORTS:
        return SERVICE_PORTS[sport]

    # UDP 且不是已知服务 → 归入 DNS 或其他
    if proto == 17:
        return 5  # 其他

    # 双高端口 → 可能是端口扫描/C2通信
    if sport is not None and dport is not None:
        if sport > 1024 and dport > 1024:
            return 4  # 高端口扫描/P2P

    return 5  # 其他


def compute_entropy(counts):
    """计算分布的香农熵"""
    total = sum(counts)
    if total == 0:
        return 0.0
    probs = [c / total for c in counts if c > 0]
    return -sum(p * math.log2(p) for p in probs)


# ---------------------------------------------------------------------------
#  从 pcap 解析（复用 darpa_pcap_to_pt 的解析逻辑）
# ---------------------------------------------------------------------------

def parse_pcap_packets(path, limit=None):
    """解析 pcap.gz 文件，生成原始数据包"""
    opener = gzip.open if str(path).endswith('.gz') else open
    f = opener(path, 'rb')
    gh = f.read(24)
    if len(gh) < 24:
        f.close()
        return
    magic = gh[:4]
    endian = '<' if magic == b'\xd4\xc3\xb2\xa1' else '>'
    linktype = struct.unpack(endian + 'I', gh[20:24])[0]
    count = 0
    while True:
        ph = f.read(16)
        if not ph or len(ph) < 16:
            break
        ts_sec, ts_usec, incl_len, orig_len = struct.unpack(endian + 'IIII', ph)
        data = f.read(incl_len)
        if not data or len(data) < incl_len:
            break
        if linktype == 1:
            yield ts_sec, data
        count += 1
        if limit and count >= limit:
            break
    f.close()


def parse_ipv4(data):
    """从以太网帧解析 IPv4 包头"""
    if len(data) < 14:
        return None
    ethtype = struct.unpack('!H', data[12:14])[0]
    pos = 14
    if ethtype == 0x8100:
        if len(data) < pos + 6:
            return None
        ethtype = struct.unpack('!H', data[pos + 4:pos + 6])[0]
        pos += 6
    if ethtype != 0x0800 or len(data) < pos + 20:
        return None
    ver_ihl = data[pos]
    ihl = (ver_ihl & 0x0F) * 4
    if ihl < 20 or len(data) < pos + ihl:
        return None
    proto = data[pos + 9]
    src_ip = '.'.join(str(x) for x in data[pos + 12:pos + 16])
    dst_ip = '.'.join(str(x) for x in data[pos + 16:pos + 20])
    l4pos = pos + ihl
    sport, dport = None, None
    if proto in (6, 17) and len(data) >= l4pos + 4:
        sport, dport = struct.unpack('!HH', data[l4pos:l4pos + 4])
    return src_ip, dst_ip, proto, sport, dport


# ---------------------------------------------------------------------------
#  核心：构建语义图谱
# ---------------------------------------------------------------------------

def build_from_pcap(pcap_paths, limit=None):
    """从 pcap 文件构建多关系语义图谱

    Returns:
        id2idx, ips, features, edge_index, edge_type, edge_weight, graph_info
    """
    print("="*60)
    print("多关系安全攻击图谱 - 语义构建")
    print("="*60)

    # --- 第1步：扫描所有包，收集统计信息 ---
    nodes = {}
    # 边统计：(src_ip, dst_ip, service_type) -> {count, bytes}
    edge_stats = defaultdict(lambda: {'count': 0, 'bytes': 0})
    # 节点详细统计
    node_detail = defaultdict(lambda: {
        'in_bytes': 0, 'out_bytes': 0, 'in_pkts': 0, 'out_pkts': 0,
        'tcp_in': 0, 'tcp_out': 0, 'udp_in': 0, 'udp_out': 0,
        'dst_ports': set(), 'src_ports': set(), 'peers': set(),
        'service_counts': [0]*6,  # 6 种服务类型的计数
        'pkt_sizes': [],
        'timestamps': [],
    })

    pkt_count = 0
    for p in pcap_paths:
        print(f"  解析: {p.name}")
        for ts, pkt in parse_pcap_packets(p, limit=limit):
            parsed = parse_ipv4(pkt)
            if not parsed:
                continue
            src_ip, dst_ip, proto, sport, dport = parsed
            if proto not in (6, 17):
                continue

            nodes[src_ip] = True
            nodes[dst_ip] = True

            # 边语义分类
            svc = classify_edge(proto, sport, dport)
            key = (src_ip, dst_ip, svc)
            edge_stats[key]['count'] += 1
            edge_stats[key]['bytes'] += len(pkt)

            # 源节点统计
            ns = node_detail[src_ip]
            ns['out_bytes'] += len(pkt)
            ns['out_pkts'] += 1
            if proto == 6:
                ns['tcp_out'] += len(pkt)
            else:
                ns['udp_out'] += len(pkt)
            if dport is not None:
                ns['dst_ports'].add(dport)
            if sport is not None:
                ns['src_ports'].add(sport)
            ns['peers'].add(dst_ip)
            ns['service_counts'][svc] += 1
            ns['pkt_sizes'].append(len(pkt))
            ns['timestamps'].append(ts)

            # 目的节点统计
            nd = node_detail[dst_ip]
            nd['in_bytes'] += len(pkt)
            nd['in_pkts'] += 1
            if proto == 6:
                nd['tcp_in'] += len(pkt)
            else:
                nd['udp_in'] += len(pkt)
            nd['peers'].add(src_ip)

            pkt_count += 1

    print(f"\n  总包数: {pkt_count:,}")
    print(f"  节点数: {len(nodes):,}")
    print(f"  语义边数: {len(edge_stats):,}")

    # --- 第2步：构建节点索引 ---
    ips = sorted(nodes.keys())
    id2idx = {ip: i for i, ip in enumerate(ips)}
    num_nodes = len(ips)

    # --- 第3步：构建边 ---
    src_list, dst_list, type_list, weight_list = [], [], [], []
    svc_edge_counts = [0] * 6
    for (s, d, svc), stats in edge_stats.items():
        src_list.append(id2idx[s])
        dst_list.append(id2idx[d])
        type_list.append(svc)
        weight_list.append(float(stats['count']))  # 权重 = 连接次数
        svc_edge_counts[svc] += 1

    edge_index = torch.tensor([src_list, dst_list], dtype=torch.long)
    edge_type = torch.tensor(type_list, dtype=torch.long)
    edge_weight = torch.tensor(weight_list, dtype=torch.float32)

    svc_names = ['HTTP/HTTPS', 'SSH/远程', 'DNS', '邮件', '高端口扫描', '其他']
    print("\n  边类型分布:")
    for i, name in enumerate(svc_names):
        print(f"    类型{i} ({name}): {svc_edge_counts[i]:,} 条边")

    # --- 第4步：构建扩展节点特征 (20维) ---
    feats = []
    for ip in ips:
        nd = node_detail[ip]
        # 原始 8 维
        f = [
            float(nd['in_bytes']), float(nd['out_bytes']),
            float(nd['in_pkts']), float(nd['out_pkts']),
            float(nd['tcp_in']), float(nd['tcp_out']),
            float(nd['udp_in']), float(nd['udp_out']),
        ]
        # 新增: 端口与连接多样性 (3维)
        f.append(float(len(nd['dst_ports'])))   # unique_dst_ports
        f.append(float(len(nd['src_ports'])))   # unique_src_ports
        f.append(float(len(nd['peers'])))        # unique_peers

        # 新增: 服务类型分布比例 (5维, 去掉"其他"避免共线)
        total_svc = sum(nd['service_counts']) or 1
        f.append(nd['service_counts'][0] / total_svc)  # http_ratio
        f.append(nd['service_counts'][1] / total_svc)  # ssh_ratio
        f.append(nd['service_counts'][2] / total_svc)  # dns_ratio
        f.append(nd['service_counts'][3] / total_svc)  # mail_ratio
        f.append(nd['service_counts'][4] / total_svc)  # scan_ratio

        # 新增: 包大小统计 (1维)
        sizes = nd['pkt_sizes']
        f.append(np.mean(sizes) if sizes else 0.0)  # avg_pkt_size

        # 新增: 时序突发率 (1维)
        ts = sorted(nd['timestamps'])
        if len(ts) > 1:
            intervals = [ts[i+1] - ts[i] for i in range(len(ts)-1)]
            # 1秒内的最大包数作为突发率
            burst = max(1.0 / (iv + 0.001) for iv in intervals)
            f.append(min(burst, 10000.0))
        else:
            f.append(0.0)

        # 新增: 流量方向比 (1维)
        total_bytes = nd['in_bytes'] + nd['out_bytes']
        f.append(nd['out_bytes'] / total_bytes if total_bytes > 0 else 0.5)

        # 新增: 端口熵 (1维)
        port_counts = list(defaultdict(int, {}).values()) or [0]
        all_ports = list(nd['dst_ports']) + list(nd['src_ports'])
        if all_ports:
            port_freq = defaultdict(int)
            for port in all_ports:
                port_freq[port] += 1
            f.append(compute_entropy(list(port_freq.values())))
        else:
            f.append(0.0)

        feats.append(f)

    features = torch.tensor(feats, dtype=torch.float32)
    print(f"\n  节点特征维度: {features.shape[1]}")

    # 图谱信息
    graph_info = {
        'num_nodes': num_nodes,
        'num_edges': edge_index.size(1),
        'num_relations': 6,
        'feature_dim': features.shape[1],
        'service_names': svc_names,
        'edge_distribution': {name: cnt for name, cnt in zip(svc_names, svc_edge_counts)},
        'feature_names': [
            'in_bytes', 'out_bytes', 'in_pkts', 'out_pkts',
            'tcp_in', 'tcp_out', 'udp_in', 'udp_out',
            'unique_dst_ports', 'unique_src_ports', 'unique_peers',
            'http_ratio', 'ssh_ratio', 'dns_ratio', 'mail_ratio', 'scan_ratio',
            'avg_pkt_size', 'max_burst_rate', 'out_byte_ratio', 'port_entropy'
        ],
    }

    return id2idx, ips, features, edge_index, edge_type, edge_weight, graph_info


def upgrade_existing(base_dir):
    """基于现有 .pt 文件，对边类型和特征进行增强

    适用于已经有 features.pt / edge_index.pt 但想扩展语义的场景。
    当没有原始 pcap 时，通过规则推断扩展边类型和特征。
    """
    base = Path(base_dir)
    print("="*60)
    print("语义增强模式 - 基于现有 .pt 文件")
    print("="*60)

    features = torch.load(base / 'features.pt')
    edge_index = torch.load(base / 'edge_index.pt')
    edge_type_old = torch.load(base / 'edge_type.pt')
    edge_weight = torch.load(base / 'edge_weight.pt') if (base / 'edge_weight.pt').exists() else torch.ones(edge_index.size(1))

    num_nodes = features.size(0)
    num_edges = edge_index.size(1)
    old_feat_dim = features.size(1)

    print(f"  节点: {num_nodes}, 边: {num_edges}, 原始特征维度: {old_feat_dim}")
    print(f"  原始边类型数: {edge_type_old.unique().numel()}")

    # --- 基于流量特征推断更细粒度的边类型 ---
    # 策略：利用节点流量模式 + 原始边类型 + 度信息，将 2 类边扩展为 6 类
    src = edge_index[0]
    dst = edge_index[1]

    # 计算每个节点的度
    deg_out = torch.zeros(num_nodes)
    deg_out.index_add_(0, src, torch.ones(num_edges))
    deg_in = torch.zeros(num_nodes)
    deg_in.index_add_(0, dst, torch.ones(num_edges))

    # 节点的出/入流量
    out_bytes = features[:, 1] if old_feat_dim > 1 else torch.zeros(num_nodes)
    in_bytes = features[:, 0] if old_feat_dim > 0 else torch.zeros(num_nodes)

    # 推断边类型规则：
    new_edge_type = torch.zeros(num_edges, dtype=torch.long)
    for i in range(num_edges):
        s, d = src[i].item(), dst[i].item()
        old_t = edge_type_old[i].item()
        w = edge_weight[i].item()

        # 高度节点间的连接 → SSH/C2 (类型1)
        if deg_out[s] > 50 and deg_out[d] > 50:
            new_edge_type[i] = 1
        # 高扇出节点 → 扫描行为 (类型4)
        elif deg_out[s] > 100 and out_bytes[s] > out_bytes.median():
            new_edge_type[i] = 4
        # UDP 流量 → DNS 相关 (类型2)
        elif old_t == 1:  # 原 UDP
            new_edge_type[i] = 2
        # 大流量 TCP → HTTP (类型0)
        elif old_t == 0 and w > edge_weight.median():
            new_edge_type[i] = 0
        # 小流量 TCP → 其他服务 (类型5)
        elif old_t == 0:
            new_edge_type[i] = 5
        else:
            new_edge_type[i] = 5

    # 统计新边类型分布
    svc_names = ['HTTP/HTTPS', 'SSH/远程', 'DNS', '邮件', '高端口扫描', '其他']
    print("\n  推断后边类型分布:")
    for i in range(6):
        cnt = (new_edge_type == i).sum().item()
        print(f"    类型{i} ({svc_names[i]}): {cnt:,} 条边")

    # --- 扩展节点特征 ---
    # 计算每个节点的入/出度、邻居数等
    unique_peers = torch.zeros(num_nodes)
    for i in range(num_edges):
        s, d = src[i].item(), dst[i].item()
        unique_peers[s] += 1  # 简化：用出边数近似唯一邻居数

    # 服务类型分布特征
    svc_counts = torch.zeros(num_nodes, 6)
    for i in range(num_edges):
        s = src[i].item()
        t = new_edge_type[i].item()
        svc_counts[s][t] += 1

    svc_total = svc_counts.sum(dim=1, keepdim=True).clamp(min=1)
    svc_ratios = svc_counts / svc_total  # [N, 6]

    # 流量方向比
    total_bytes = in_bytes + out_bytes
    out_ratio = torch.where(total_bytes > 0, out_bytes / total_bytes, torch.tensor(0.5))

    # 组合新特征
    extra_feats = torch.stack([
        deg_out.float(),              # unique_dst_ports 近似
        deg_in.float(),               # unique_src_ports 近似
        unique_peers,                 # unique_peers
        svc_ratios[:, 0],            # http_ratio
        svc_ratios[:, 1],            # ssh_ratio
        svc_ratios[:, 2],            # dns_ratio
        svc_ratios[:, 3],            # mail_ratio
        svc_ratios[:, 4],            # scan_ratio
        (out_bytes / out_bytes.clamp(min=1).max()),  # avg_pkt_size 近似
        deg_out / deg_out.clamp(min=1).max(),        # burst_rate 近似
        out_ratio,                    # out_byte_ratio
        torch.zeros(num_nodes),       # port_entropy 占位
    ], dim=1)

    new_features = torch.cat([features, extra_feats], dim=1)
    print(f"\n  新特征维度: {old_feat_dim} → {new_features.size(1)}")

    return new_features, edge_index, new_edge_type, edge_weight, {
        'num_nodes': num_nodes,
        'num_edges': num_edges,
        'num_relations': 6,
        'feature_dim': new_features.size(1),
    }


# ---------------------------------------------------------------------------
#  主函数
# ---------------------------------------------------------------------------

def main():
    args = sys.argv

    def get_arg(name, default=None):
        if name in args:
            try:
                return args[args.index(name) + 1]
            except IndexError:
                return default
        return default

    # 模式1: 从 pcap 构建
    input_dir = get_arg('--input_dir')
    output_dir = get_arg('--output_dir', input_dir or str(Path(__file__).parent))
    limit = get_arg('--limit')
    limit = int(limit) if limit else None

    # 模式2: 基于现有 .pt 文件升级
    if '--upgrade' in args:
        dataset_dir = get_arg('--dataset', str(Path(__file__).parent))
        out = Path(get_arg('--output_dir', dataset_dir))

        features, edge_index, edge_type, edge_weight, info = upgrade_existing(dataset_dir)

        # 备份原文件
        backup_dir = out / 'backup_2rel'
        backup_dir.mkdir(exist_ok=True)
        for name in ['features.pt', 'edge_index.pt', 'edge_type.pt', 'edge_weight.pt']:
            src_path = out / name
            if src_path.exists():
                torch.save(torch.load(src_path), backup_dir / name)

        # 保存增强后的图谱
        torch.save(features, out / 'features.pt')
        torch.save(edge_index, out / 'edge_index.pt')
        torch.save(edge_type, out / 'edge_type.pt')
        torch.save(edge_weight, out / 'edge_weight.pt')

        # 保存图谱信息
        info_text = '\n'.join(f'{k}={v}' for k, v in info.items())
        (out / 'graph_info.txt').write_text(info_text, encoding='utf-8')

        print(f"\n  已保存到: {out}")
        print(f"  原始文件备份到: {backup_dir}")
        print("\n  完成！现在可以用新图谱训练：")
        print(f"    python train_mrgnn.py --model rgcn_group --epochs 50")
        return

    # 模式1: 从 pcap 构建
    if input_dir is None:
        print("用法:")
        print("  从 pcap 构建:  python build_semantic_graph.py --input_dir <pcap目录>")
        print("  升级现有图谱:  python build_semantic_graph.py --upgrade [--dataset <目录>]")
        return

    in_dir = Path(input_dir)
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    paths = []
    for name in ['outside.tcpdump.gz', 'inside.tcpdump.gz',
                 'outside.tcpdump', 'inside.tcpdump']:
        p = in_dir / name
        if p.exists():
            paths.append(p)

    if not paths:
        print(f"错误: 在 {in_dir} 中未找到 pcap 文件")
        return

    id2idx, ips, features, edge_index, edge_type, edge_weight, info = build_from_pcap(paths, limit)

    torch.save(features, out / 'features.pt')
    torch.save(edge_index, out / 'edge_index.pt')
    torch.save(edge_type, out / 'edge_type.pt')
    torch.save(edge_weight, out / 'edge_weight.pt')
    (out / 'ip_index.txt').write_text(
        '\n'.join(f'{i},{ip}' for ip, i in id2idx.items()),
        encoding='utf-8'
    )

    import json
    (out / 'graph_info.json').write_text(
        json.dumps(info, ensure_ascii=False, indent=2),
        encoding='utf-8'
    )

    print(f"\n  图谱已保存到: {out}")
    print(f"  训练命令: python train_mrgnn.py --model rgcn_group --epochs 50")


if __name__ == '__main__':
    main()
