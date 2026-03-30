import sys
import gzip
import struct
from pathlib import Path
import torch

def parse_pcap_packets(path, limit=None):
    f = gzip.open(path, 'rb')
    gh = f.read(24)
    if len(gh) < 24:
        f.close()
        return
    magic = gh[:4]
    if magic == b'\xd4\xc3\xb2\xa1':
        endian = '<'
    elif magic == b'\xa1\xb2\xc3\xd4':
        endian = '>'
    else:
        endian = '<'
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
            yield data
        count += 1
        if limit is not None and count >= limit:
            break
    f.close()

def parse_ether_ipv4(data, offset=0):
    if len(data) < offset + 14:
        return None
    ethtype = struct.unpack('!H', data[offset+12:offset+14])[0]
    pos = offset + 14
    if ethtype == 0x8100:
        if len(data) < pos + 4 + 2:
            return None
        ethtype = struct.unpack('!H', data[pos+4:pos+6])[0]
        pos += 4 + 2
    if ethtype != 0x0800:
        return None
    if len(data) < pos + 20:
        return None
    ver_ihl = data[pos]
    ihl = (ver_ihl & 0x0F) * 4
    if ihl < 20 or len(data) < pos + ihl:
        return None
    proto = data[pos+9]
    src = data[pos+12:pos+16]
    dst = data[pos+16:pos+20]
    src_ip = '.'.join(str(x) for x in src)
    dst_ip = '.'.join(str(x) for x in dst)
    l4pos = pos + ihl
    sport = None
    dport = None
    if proto == 6:
        if len(data) >= l4pos + 4:
            sport, dport = struct.unpack('!HH', data[l4pos:l4pos+4])
    elif proto == 17:
        if len(data) >= l4pos + 4:
            sport, dport = struct.unpack('!HH', data[l4pos:l4pos+4])
    return src_ip, dst_ip, proto, sport, dport

def aggregate_edges(paths, limit=None):
    edges = {}
    nodes = {}
    node_stats = {}
    for p in paths:
        for pkt in parse_pcap_packets(p, limit=limit):
            parsed = parse_ether_ipv4(pkt, 0)
            if not parsed:
                continue
            src_ip, dst_ip, proto, sport, dport = parsed
            if proto not in (6, 17):
                continue
            nodes[src_ip] = True
            nodes[dst_ip] = True
            key = (src_ip, dst_ip, proto)
            w = edges.get(key)
            if w is None:
                edges[key] = len(pkt)
            else:
                edges[key] = w + len(pkt)
            ns = node_stats.get(src_ip)
            if ns is None:
                ns = {'in_bytes': 0, 'in_pkts': 0, 'tcp_in_bytes': 0, 'udp_in_bytes': 0, 'out_bytes': 0, 'out_pkts': 0, 'tcp_out_bytes': 0, 'udp_out_bytes': 0}
            ns['out_bytes'] += len(pkt)
            ns['out_pkts'] += 1
            if proto == 6:
                ns['tcp_out_bytes'] += len(pkt)
            else:
                ns['udp_out_bytes'] += len(pkt)
            node_stats[src_ip] = ns
            nd = node_stats.get(dst_ip)
            if nd is None:
                nd = {'in_bytes': 0, 'in_pkts': 0, 'tcp_in_bytes': 0, 'udp_in_bytes': 0, 'out_bytes': 0, 'out_pkts': 0, 'tcp_out_bytes': 0, 'udp_out_bytes': 0}
            nd['in_bytes'] += len(pkt)
            nd['in_pkts'] += 1
            if proto == 6:
                nd['tcp_in_bytes'] += len(pkt)
            else:
                nd['udp_in_bytes'] += len(pkt)
            node_stats[dst_ip] = nd
    ips = sorted(nodes.keys())
    id2idx = {ip: i for i, ip in enumerate(ips)}
    src_list = []
    dst_list = []
    type_list = []
    weight_list = []
    for (s, d, proto), w in edges.items():
        src_list.append(id2idx[s])
        dst_list.append(id2idx[d])
        type_list.append(0 if proto == 6 else 1)
        weight_list.append(float(w))
    edge_index = torch.tensor([src_list, dst_list], dtype=torch.long)
    edge_type = torch.tensor(type_list, dtype=torch.long)
    edge_weight = torch.tensor(weight_list, dtype=torch.float32)
    feats = []
    for ip in ips:
        st = node_stats.get(ip, {})
        in_bytes = float(st.get('in_bytes', 0))
        out_bytes = float(st.get('out_bytes', 0))
        in_pkts = float(st.get('in_pkts', 0))
        out_pkts = float(st.get('out_pkts', 0))
        tcp_in = float(st.get('tcp_in_bytes', 0))
        tcp_out = float(st.get('tcp_out_bytes', 0))
        udp_in = float(st.get('udp_in_bytes', 0))
        udp_out = float(st.get('udp_out_bytes', 0))
        feats.append([in_bytes, out_bytes, in_pkts, out_pkts, tcp_in, tcp_out, udp_in, udp_out])
    features = torch.tensor(feats, dtype=torch.float32)
    return id2idx, ips, features, edge_index, edge_type, edge_weight

def main():
    if '--input_dir' not in sys.argv:
        return
    in_dir = Path(sys.argv[sys.argv.index('--input_dir') + 1])
    out_dir = in_dir
    if '--output_dir' in sys.argv:
        out_dir = Path(sys.argv[sys.argv.index('--output_dir') + 1])
    limit = None
    if '--limit' in sys.argv:
        try:
            limit = int(sys.argv[sys.argv.index('--limit') + 1])
        except:
            limit = None
    paths = []
    p1 = in_dir / 'outside.tcpdump.gz'
    p2 = in_dir / 'inside.tcpdump.gz'
    if p1.exists():
        paths.append(p1)
    if p2.exists():
        paths.append(p2)
    id2idx, ips, features, edge_index, edge_type, edge_weight = aggregate_edges(paths, limit=limit)
    torch.save(features, out_dir / 'features.pt')
    torch.save(edge_index, out_dir / 'edge_index.pt')
    torch.save(edge_type, out_dir / 'edge_type.pt')
    torch.save(edge_weight, out_dir / 'edge_weight.pt')
    Path(out_dir / 'ip_index.txt').write_text('\n'.join(f'{i},{ip}' for ip, i in id2idx.items()), encoding='utf-8')

if __name__ == '__main__':
    main()
