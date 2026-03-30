"""GE-MRGNN: 群组增强的多关系图神经网络

基于R-GCN (Schlichtkrull et al., 2018) 和 GraphSAGE 的邻居聚合思想，
针对网络安全团伙检测场景设计的图神经网络模型。

核心模块：
- RGCNLayer: 多关系图卷积层，为不同类型的边学习独立的变换矩阵
- GroupEnhance: 群组增强模块，通过邻居聚合增强节点的群组感知能力
- MR_GNN: 完整的模型框架，支持多种变体 (gcn/gat/rgcn/rgcn_group)
"""
import sys
from pathlib import Path
import torch
import torch.nn as nn
import torch.nn.functional as F
import random
import numpy as np


def set_seed(seed):
    """设置随机种子以确保可复现性"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


class RGCNLayer(nn.Module):
    """关系图卷积层 (Relational Graph Convolutional Layer)
    
    为每种关系类型学习独立的变换矩阵W_r，实现多关系建模。
    参考论文: Modeling Relational Data with Graph Convolutional Networks (Schlichtkrull et al., 2018)
    """
    def __init__(self, in_dim, out_dim, num_relations, use_root=True):
        super().__init__()
        self.use_root = use_root
        self.out_dim = out_dim
        # 每种关系一个权重矩阵
        self.W = nn.Parameter(torch.randn(num_relations, in_dim, out_dim) * 0.02)
        if use_root:
            # 自环权重（节点自身特征的变换）
            self.W0 = nn.Parameter(torch.randn(in_dim, out_dim) * 0.02)

    def forward(self, x, edge_index, edge_type, num_nodes):
        device = x.device
        out = torch.zeros((num_nodes, self.out_dim), device=device)
        if self.use_root:
            out = out + x @ self.W0
        rels = torch.unique(edge_type)
        for r in rels.tolist():
            mask = (edge_type == r)
            src = edge_index[0][mask]
            dst = edge_index[1][mask]
            xW = x @ self.W[r]
            out.index_add_(0, dst, xW[src])
        # 度归一化
        deg = torch.zeros((num_nodes,), device=device)
        deg.index_add_(0, edge_index[1], torch.ones_like(edge_index[1], dtype=deg.dtype, device=device))
        deg = deg.clamp(min=1.0)
        out = out / deg.unsqueeze(1)
        return out


class GCNLayer(nn.Module):
    def __init__(self, in_dim, out_dim, use_root=True):
        super().__init__()
        self.use_root = use_root
        self.W = nn.Parameter(torch.randn(in_dim, out_dim) * 0.02)
        if use_root:
            self.W0 = nn.Parameter(torch.randn(in_dim, out_dim) * 0.02)

    def forward(self, x, edge_index, num_nodes):
        device = x.device
        out = torch.zeros((num_nodes, self.W.shape[-1]), device=device)
        if self.use_root:
            out = out + x @ self.W0
        src = edge_index[0]
        dst = edge_index[1]
        xW = x @ self.W
        out.index_add_(0, dst, xW[src])
        deg = torch.zeros((num_nodes,), device=device)
        deg.index_add_(0, dst, torch.ones_like(dst, dtype=deg.dtype, device=device))
        deg = deg.clamp(min=1.0)
        out = out / deg.unsqueeze(1)
        return out

class GroupEnhance(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.alpha = nn.Parameter(torch.tensor(0.5))
        self.proj = nn.Linear(dim, dim)

    def forward(self, x, edge_index, num_nodes):
        device = x.device
        agg = torch.zeros_like(x, device=device)
        deg = torch.zeros((num_nodes,), device=device)
        src = edge_index[0]
        dst = edge_index[1]
        agg.index_add_(0, dst, x[src])
        deg.index_add_(0, dst, torch.ones_like(dst, dtype=deg.dtype, device=device))
        deg = deg.clamp(min=1.0)
        agg = agg / deg.unsqueeze(1)
        return x + self.alpha * self.proj(agg)

class GATLayer(nn.Module):
    def __init__(self, in_dim, out_dim):
        super().__init__()
        self.W = nn.Linear(in_dim, out_dim, bias=False)
        self.a = nn.Parameter(torch.randn(out_dim * 2) * 0.02)
        self.leaky = nn.LeakyReLU(0.2)

    def forward(self, x, edge_index, num_nodes):
        h = self.W(x)
        src = edge_index[0]
        dst = edge_index[1]
        hs = h[src]
        hd = h[dst]
        concat = torch.cat([hs, hd], dim=1)
        e = self.leaky(concat @ self.a)
        out = torch.zeros_like(h)
        unique_dst = torch.unique(dst)
        for d in unique_dst.tolist():
            mask = (dst == d)
            scores = e[mask]
            alpha = torch.softmax(scores, dim=0)
            agg = (hs[mask] * alpha.unsqueeze(1)).sum(dim=0)
            out[d] = agg
        return out

class MR_GNN(nn.Module):
    def __init__(self, in_dim, hidden_dim, out_dim, num_relations, model_type='rgcn_group'):
        super().__init__()
        self.model_type = model_type
        if model_type == 'gcn':
            self.gcn1 = GCNLayer(in_dim, hidden_dim)
            self.gcn2 = GCNLayer(hidden_dim, hidden_dim)
            self.out = nn.Linear(hidden_dim, out_dim)
        elif model_type == 'gat':
            self.gat1 = GATLayer(in_dim, hidden_dim)
            self.gat2 = GATLayer(hidden_dim, hidden_dim)
            self.out = nn.Linear(hidden_dim, out_dim)
        elif model_type == 'rgcn':
            self.rgcn1 = RGCNLayer(in_dim, hidden_dim, num_relations)
            self.rgcn2 = RGCNLayer(hidden_dim, hidden_dim, num_relations)
            self.out = nn.Linear(hidden_dim, out_dim)
        else:
            self.rgcn1 = RGCNLayer(in_dim, hidden_dim, num_relations)
            self.group1 = GroupEnhance(hidden_dim)
            self.rgcn2 = RGCNLayer(hidden_dim, hidden_dim, num_relations)
            self.group2 = GroupEnhance(hidden_dim)
            self.out = nn.Linear(hidden_dim, out_dim)

    def forward(self, x, edge_index, edge_type):
        num_nodes = x.size(0)
        if self.model_type == 'gcn':
            h = self.gcn1(x, edge_index, num_nodes)
            h = F.relu(h)
            h = self.gcn2(h, edge_index, num_nodes)
            h = F.relu(h)
            logits = self.out(h)
            return logits
        elif self.model_type == 'gat':
            h = self.gat1(x, edge_index, num_nodes)
            h = F.relu(h)
            h = self.gat2(h, edge_index, num_nodes)
            h = F.relu(h)
            logits = self.out(h)
            return logits
        elif self.model_type == 'rgcn':
            h = self.rgcn1(x, edge_index, edge_type, num_nodes)
            h = F.relu(h)
            h = self.rgcn2(h, edge_index, edge_type, num_nodes)
            h = F.relu(h)
            logits = self.out(h)
            return logits
        else:
            h = self.rgcn1(x, edge_index, edge_type, num_nodes)
            h = F.relu(h)
            h = self.group1(h, edge_index, num_nodes)
            h = self.rgcn2(h, edge_index, edge_type, num_nodes)
            h = F.relu(h)
            h = self.group2(h, edge_index, num_nodes)
            logits = self.out(h)
            return logits

def split_dataset(labels, train_ratio=0.6, val_ratio=0.2, seed=42):
    """将有标签的节点划分为训练/验证/测试集
    
    Args:
        labels: 节点标签，-1表示无标签
        train_ratio: 训练集比例
        val_ratio: 验证集比例
        seed: 随机种子
    
    Returns:
        train_mask, val_mask, test_mask
    """
    num_nodes = labels.size(0)
    labeled_idx = (labels >= 0).nonzero(as_tuple=True)[0]
    
    # 打乱顺序
    perm = torch.randperm(len(labeled_idx), generator=torch.Generator().manual_seed(seed))
    labeled_idx = labeled_idx[perm]
    
    n_train = int(len(labeled_idx) * train_ratio)
    n_val = int(len(labeled_idx) * val_ratio)
    
    train_idx = labeled_idx[:n_train]
    val_idx = labeled_idx[n_train:n_train + n_val]
    test_idx = labeled_idx[n_train + n_val:]
    
    train_mask = torch.zeros(num_nodes, dtype=torch.bool)
    val_mask = torch.zeros(num_nodes, dtype=torch.bool)
    test_mask = torch.zeros(num_nodes, dtype=torch.bool)
    
    train_mask[train_idx] = True
    val_mask[val_idx] = True
    test_mask[test_idx] = True
    
    return train_mask, val_mask, test_mask


def load_dataset(dir_path):
    base = Path(dir_path)
    features = torch.load(base / 'features.pt')
    edge_index = torch.load(base / 'edge_index.pt')
    edge_type = torch.load(base / 'edge_type.pt')
    labels_bot = None
    labels_stance = None
    p_bot = base / 'labels_bot.pt'
    p_stance = base / 'labels_stance.pt'
    if p_bot.exists():
        labels_bot = torch.load(p_bot)
    if p_stance.exists():
        labels_stance = torch.load(p_stance)
    return features, edge_index, edge_type, labels_bot, labels_stance

def main():
    args = sys.argv
    if '--dataset' in args:
        dataset_dir = Path(args[args.index('--dataset') + 1])
    else:
        dataset_dir = Path(__file__).parent
    task = 'bot'
    if '--task' in args:
        task = args[args.index('--task') + 1]
    epochs = 100  # 默认100轮
    if '--epochs' in args:
        epochs = int(args[args.index('--epochs') + 1])
    hidden = 64
    if '--hidden' in args:
        hidden = int(args[args.index('--hidden') + 1])
    lr = 0.001
    if '--lr' in args:
        lr = float(args[args.index('--lr') + 1])
    patience = 20  # early stopping 耐心值
    if '--patience' in args:
        patience = int(args[args.index('--patience') + 1])
    max_edges = None
    if '--max_edges' in args:
        max_edges = int(args[args.index('--max_edges') + 1])
    model_type = 'rgcn_group'
    if '--model' in args:
        model_type = args[args.index('--model') + 1]
    show_top = 0
    if '--show_top' in args:
        show_top = int(args[args.index('--show_top') + 1])
    out_dir = None
    if '--out_dir' in args:
        out_dir = Path(args[args.index('--out_dir') + 1])
    seed = None
    if '--seed' in args:
        try:
            seed = int(args[args.index('--seed') + 1])
        except:
            seed = None
    if seed is not None:
        set_seed(seed)
    features, edge_index, edge_type, labels_bot, labels_stance = load_dataset(dataset_dir)
    num_nodes = features.size(0)
    if max_edges is not None and edge_index.size(1) > max_edges:
        idx = torch.arange(edge_index.size(1))[:max_edges]
        edge_index = edge_index[:, idx]
        edge_type = edge_type[idx]
    num_rel = int(edge_type.max().item()) + 1 if edge_type.numel() > 0 else 1
    if task == 'bot' and labels_bot is not None:
        labels = labels_bot
        out_dim = 1
        pos_weight_val = None
        if '--pos_weight' in args:
            try:
                pos_weight_val = float(args[args.index('--pos_weight') + 1])
            except:
                pos_weight_val = None
        if pos_weight_val is None:
            pos = (labels == 1).sum().item()
            neg = (labels == 0).sum().item()
            pos_weight_val = (neg / max(1, pos)) if (pos + neg) > 0 else 1.0
        criterion = nn.BCEWithLogitsLoss(pos_weight=torch.tensor(pos_weight_val))
        is_multiclass = False
    elif task == 'stance' and labels_stance is not None:
        labels = labels_stance
        out_dim = int(labels.max().item()) + 1
        criterion = nn.CrossEntropyLoss()
        is_multiclass = True
    else:
        print("未找到有效标签文件")
        return
    
    # 划分训练/验证/测试集
    train_mask, val_mask, test_mask = split_dataset(labels, seed=seed if seed else 42)
    print(f"数据集划分: 训练={train_mask.sum().item()}, 验证={val_mask.sum().item()}, 测试={test_mask.sum().item()}")
    
    mask = labels >= 0  # 所有有标签的节点
    if model_type == 'gcn':
        edge_type = torch.zeros_like(edge_type)
        num_rel = 1
    if model_type == 'lpa':
        adj = [[] for _ in range(num_nodes)]
        for i in range(edge_index.size(1)):
            s = int(edge_index[0, i])
            d = int(edge_index[1, i])
            adj[d].append(s)
        pred = labels.clone()
        for _ in range(5):
            for u in range(num_nodes):
                if pred[u] < 0 and len(adj[u]) > 0:
                    vs = torch.tensor([pred[v] for v in adj[u] if pred[v] >= 0], dtype=torch.long)
                    if vs.numel() > 0:
                        val = int((vs.float().mean() > 0.5).item()) if not is_multiclass else int(torch.mode(vs).values.item())
                        pred[u] = val
        acc = (pred[mask] == labels[mask]).float().mean().item()
        print(f'lpa acc={acc:.4f}')
        return
    model = MR_GNN(features.size(1), hidden, out_dim, num_rel, model_type=model_type)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-5)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=10)
    
    # Early stopping
    best_val_loss = float('inf')
    best_model_state = None
    patience_counter = 0
    
    # 模型名称映射，让输出更清晰
    model_names = {
        'rgcn_group': 'GE-MRGNN (群组增强的多关系图神经网络)',
        'rgcn': 'RGCN (多关系图卷积网络)',
        'gcn': 'GCN (图卷积网络)',
        'gat': 'GAT (图注意力网络)',
        'lpa': 'LPA (标签传播算法)'
    }
    display_name = model_names.get(model_type, model_type.upper())
    print(f"\n开始训练 {display_name}...")
    print(f"参数: hidden={hidden}, lr={lr}, epochs={epochs}, patience={patience}")
    print("-" * 60)
    
    for ep in range(epochs):
        # === 训练阶段 ===
        model.train()
        logits = model(features, edge_index, edge_type)
        if is_multiclass:
            loss = criterion(logits[train_mask], labels[train_mask])
            pred = logits.argmax(dim=1)
            train_acc = (pred[train_mask] == labels[train_mask]).float().mean().item()
        else:
            loss = criterion(logits[train_mask].squeeze(1), labels[train_mask].float())
            pred = (logits.sigmoid().squeeze(1) > 0.5).long()
            train_acc = (pred[train_mask] == labels[train_mask]).float().mean().item()
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        # === 验证阶段 ===
        model.eval()
        with torch.no_grad():
            logits = model(features, edge_index, edge_type)
            if is_multiclass:
                val_loss = criterion(logits[val_mask], labels[val_mask]).item()
                val_pred = logits.argmax(dim=1)
                val_acc = (val_pred[val_mask] == labels[val_mask]).float().mean().item()
            else:
                val_loss = criterion(logits[val_mask].squeeze(1), labels[val_mask].float()).item()
                val_pred = (logits.sigmoid().squeeze(1) > 0.5).long()
                val_acc = (val_pred[val_mask] == labels[val_mask]).float().mean().item()
        
        scheduler.step(val_loss)
        
        # Early stopping 检查
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_model_state = {k: v.clone() for k, v in model.state_dict().items()}
            patience_counter = 0
        else:
            patience_counter += 1
        
        if (ep + 1) % 10 == 0 or ep == 0:
            print(f'Epoch {ep+1:3d}/{epochs} | Train Loss: {loss.item():.4f} | Train Acc: {train_acc:.4f} | Val Loss: {val_loss:.4f} | Val Acc: {val_acc:.4f}')
        
        if patience_counter >= patience:
            print(f'\nEarly stopping at epoch {ep+1}')
            break
    
    # 加载最佳模型
    if best_model_state is not None:
        model.load_state_dict(best_model_state)
    
    # === 测试阶段 ===
    model.eval()
    with torch.no_grad():
        logits = model(features, edge_index, edge_type)
        if is_multiclass:
            test_pred = logits.argmax(dim=1)
            test_acc = (test_pred[test_mask] == labels[test_mask]).float().mean().item()
        else:
            test_pred = (logits.sigmoid().squeeze(1) > 0.5).long()
            test_acc = (test_pred[test_mask] == labels[test_mask]).float().mean().item()
    
    print("-" * 60)
    print(f"测试集准确率: {test_acc:.4f}")
    print("-" * 60)
    if show_top > 0:
        with torch.no_grad():
            logits = model(features, edge_index, edge_type)
            if is_multiclass:
                probs = torch.softmax(logits, dim=1)
                conf, pred_cls = probs.max(dim=1)
                order = torch.argsort(conf, descending=True)[:show_top]
                print("Top预测示例:")
                for i in order.tolist():
                    lbl = labels[i].item()
                    print(f'节点 {i} 预测类={pred_cls[i].item()} 置信度={conf[i].item():.4f} 标签={lbl}')
            else:
                probs = torch.sigmoid(logits.squeeze(1))
                order = torch.argsort(probs, descending=True)[:show_top]
                print("Top预测示例:")
                for i in order.tolist():
                    lbl = labels[i].item()
                    print(f'节点 {i} 恶意概率={probs[i].item():.4f} 标签={lbl}')
    if out_dir is not None:
        out_dir.mkdir(parents=True, exist_ok=True)
        # 保存模型
        model_path = out_dir / f'{model_type}_model.pt'
        torch.save({
            'model_state_dict': model.state_dict(),
            'model_type': model_type,
            'hidden_dim': hidden,
            'num_relations': num_rel,
            'in_dim': features.size(1),
            'out_dim': out_dim,
            'test_acc': test_acc,
        }, model_path)
        print(f"模型已保存到: {model_path}")
        
        with torch.no_grad():
            logits = model(features, edge_index, edge_type)
            if is_multiclass:
                pred = logits.argmax(dim=1)
                (out_dir / 'summary.txt').write_text(
                    f'model={model_type}\n'
                    f'train_acc={train_acc:.6f}\n'
                    f'val_acc={val_acc:.6f}\n'
                    f'test_acc={test_acc:.6f}\n',
                    encoding='utf-8'
                )
            else:
                probs = torch.sigmoid(logits.squeeze(1))
                y = labels.float()
                y_hat = probs
                y_m = y[test_mask]  # 使用测试集计算指标
                yhat_m = y_hat[test_mask]
                th = torch.linspace(0, 1, 101)
                tpr = []
                fpr = []
                prec = []
                rec = []
                for t in th:
                    pred = (yhat_m >= t).long()
                    tp = ((pred == 1) & (y_m == 1)).sum().item()
                    fp = ((pred == 1) & (y_m == 0)).sum().item()
                    fn = ((pred == 0) & (y_m == 1)).sum().item()
                    tn = ((pred == 0) & (y_m == 0)).sum().item()
                    tpr.append(tp / (tp + fn) if (tp + fn) > 0 else 0.0)
                    fpr.append(fp / (fp + tn) if (fp + tn) > 0 else 0.0)
                    rec.append(tp / (tp + fn) if (tp + fn) > 0 else 0.0)
                    prec.append(tp / (tp + fp) if (tp + fp) > 0 else 1.0)
                roc = sorted(zip(fpr, tpr), key=lambda x: x[0])
                auc = 0.0
                for i in range(1, len(roc)):
                    x0, y0 = roc[i - 1]
                    x1, y1 = roc[i]
                    auc += (x1 - x0) * (y0 + y1) * 0.5
                pr_pts = list(zip(rec, prec))
                pr_auc = 0.0
                pr_sorted = sorted(pr_pts, key=lambda x: x[0])
                for i in range(1, len(pr_sorted)):
                    r0, p0 = pr_sorted[i - 1]
                    r1, p1 = pr_sorted[i]
                    pr_auc += (r1 - r0) * (p0 + p1) * 0.5
                roc_lines = ["threshold,fpr,tpr"]
                for i in range(len(th)):
                    roc_lines.append(f"{th[i].item():.4f},{fpr[i]:.6f},{tpr[i]:.6f}")
                pr_lines = ["threshold,precision,recall"]
                for i in range(len(th)):
                    pr_lines.append(f"{th[i].item():.4f},{prec[i]:.6f},{rec[i]:.6f}")
                (out_dir / 'roc.csv').write_text("\n".join(roc_lines), encoding='utf-8')
                (out_dir / 'pr.csv').write_text("\n".join(pr_lines), encoding='utf-8')
                (out_dir / 'summary.txt').write_text(
                    f'model={model_type}\n'
                    f'train_acc={train_acc:.6f}\n'
                    f'val_acc={val_acc:.6f}\n'
                    f'test_acc={test_acc:.6f}\n'
                    f'roc_auc={auc:.6f}\n'
                    f'pr_auc={pr_auc:.6f}\n',
                    encoding='utf-8'
                )

if __name__ == '__main__':
    main()
