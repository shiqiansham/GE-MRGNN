"""GE-MRGNN: 群组增强的多关系图神经网络

基于R-GCN (Schlichtkrull et al., 2018) 和 GraphSAGE 的邻居聚合思想，
针对网络安全团伙检测场景设计的图神经网络模型。

核心模块：
- RGCNLayer: 多关系图卷积层，为不同类型的边学习独立的变换矩阵
- GroupEnhance: 群组增强模块，通过邻居聚合增强节点的群组感知能力
- GATLayer: 图注意力层（向量化实现，无Python循环）
- HANLayer: 异质注意力网络层 (Hierarchical Attention Network)
- DGI: 深度图信息最大化 (Deep Graph Infomax) 自监督基线
- MR_GNN: 完整的模型框架，支持多种变体
"""
import sys
from pathlib import Path
import torch
import torch.nn as nn
import torch.nn.functional as F
import random
import numpy as np

# sklearn 用于更稳健的指标计算
try:
    from sklearn.metrics import (
        roc_auc_score, average_precision_score, precision_recall_curve,
        roc_curve, precision_score, recall_score, f1_score,
        confusion_matrix, classification_report
    )
    HAS_SKLEARN = True
except ImportError:
    HAS_SKLEARN = False


def set_seed(seed):
    """设置随机种子以确保可复现性"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


# ---------------------------------------------------------------------------
#  Layer 实现
# ---------------------------------------------------------------------------

class RGCNLayer(nn.Module):
    """关系图卷积层 – 支持 edge_weight 加权消息传递

    为每种关系类型学习独立的变换矩阵 W_r。
    当 edge_weight 不为 None 时，消息 = edge_weight * (x[src] @ W_r)。
    """
    def __init__(self, in_dim, out_dim, num_relations, use_root=True):
        super().__init__()
        self.use_root = use_root
        self.out_dim = out_dim
        self.W = nn.Parameter(torch.randn(num_relations, in_dim, out_dim) * 0.02)
        if use_root:
            self.W0 = nn.Parameter(torch.randn(in_dim, out_dim) * 0.02)

    def forward(self, x, edge_index, edge_type, num_nodes, edge_weight=None):
        device = x.device
        out = torch.zeros((num_nodes, self.out_dim), device=device)
        if self.use_root:
            out = out + x @ self.W0
        rels = torch.unique(edge_type)
        for r in rels.tolist():
            mask = (edge_type == r)
            src = edge_index[0][mask]
            dst = edge_index[1][mask]
            msg = x[src] @ self.W[r]
            if edge_weight is not None:
                msg = msg * edge_weight[mask].unsqueeze(1)
            out.index_add_(0, dst, msg)
        # 度归一化
        deg = torch.zeros((num_nodes,), device=device)
        deg.index_add_(0, edge_index[1], torch.ones(edge_index.size(1), device=device))
        deg = deg.clamp(min=1.0)
        out = out / deg.unsqueeze(1)
        return out


class GCNLayer(nn.Module):
    """标准图卷积层 – 支持 edge_weight"""
    def __init__(self, in_dim, out_dim, use_root=True):
        super().__init__()
        self.use_root = use_root
        self.W = nn.Parameter(torch.randn(in_dim, out_dim) * 0.02)
        if use_root:
            self.W0 = nn.Parameter(torch.randn(in_dim, out_dim) * 0.02)

    def forward(self, x, edge_index, num_nodes, edge_weight=None):
        device = x.device
        out = torch.zeros((num_nodes, self.W.shape[-1]), device=device)
        if self.use_root:
            out = out + x @ self.W0
        src = edge_index[0]
        dst = edge_index[1]
        msg = x[src] @ self.W
        if edge_weight is not None:
            msg = msg * edge_weight.unsqueeze(1)
        out.index_add_(0, dst, msg)
        deg = torch.zeros((num_nodes,), device=device)
        deg.index_add_(0, dst, torch.ones(dst.size(0), device=device))
        deg = deg.clamp(min=1.0)
        out = out / deg.unsqueeze(1)
        return out


class GroupEnhance(nn.Module):
    """群组增强模块 – 支持 edge_weight"""
    def __init__(self, dim):
        super().__init__()
        self.alpha = nn.Parameter(torch.tensor(0.5))
        self.proj = nn.Linear(dim, dim)

    def forward(self, x, edge_index, num_nodes, edge_weight=None):
        device = x.device
        agg = torch.zeros_like(x, device=device)
        src = edge_index[0]
        dst = edge_index[1]
        msg = x[src]
        if edge_weight is not None:
            msg = msg * edge_weight.unsqueeze(1)
        agg.index_add_(0, dst, msg)
        deg = torch.zeros((num_nodes,), device=device)
        deg.index_add_(0, dst, torch.ones(dst.size(0), device=device))
        deg = deg.clamp(min=1.0)
        agg = agg / deg.unsqueeze(1)
        return x + self.alpha * self.proj(agg)


class GATLayer(nn.Module):
    """图注意力层 – 完全向量化实现（无 Python for 循环）

    使用 scatter softmax 代替逐节点循环，支持大规模图。
    """
    def __init__(self, in_dim, out_dim, num_heads=1):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = out_dim // num_heads
        self.W = nn.Linear(in_dim, self.head_dim * num_heads, bias=False)
        self.a_src = nn.Parameter(torch.randn(num_heads, self.head_dim) * 0.02)
        self.a_dst = nn.Parameter(torch.randn(num_heads, self.head_dim) * 0.02)
        self.leaky = nn.LeakyReLU(0.2)

    def forward(self, x, edge_index, num_nodes, edge_weight=None):
        # x: [N, in_dim]
        h = self.W(x).view(-1, self.num_heads, self.head_dim)  # [N, H, D]
        src = edge_index[0]
        dst = edge_index[1]
        # 注意力分数
        e_src = (h[src] * self.a_src.unsqueeze(0)).sum(-1)  # [E, H]
        e_dst = (h[dst] * self.a_dst.unsqueeze(0)).sum(-1)  # [E, H]
        e = self.leaky(e_src + e_dst)  # [E, H]
        # scatter softmax: 按 dst 分组做 softmax
        e_max = torch.zeros(num_nodes, self.num_heads, device=x.device).fill_(-1e9)
        e_max.scatter_reduce_(0, dst.unsqueeze(1).expand_as(e), e, reduce='amax', include_self=False)
        e = torch.exp(e - e_max[dst])
        if edge_weight is not None:
            e = e * edge_weight.unsqueeze(1)
        e_sum = torch.zeros(num_nodes, self.num_heads, device=x.device)
        e_sum.scatter_add_(0, dst.unsqueeze(1).expand_as(e), e)
        alpha = e / (e_sum[dst] + 1e-12)  # [E, H]
        # 消息聚合
        msg = h[src] * alpha.unsqueeze(-1)  # [E, H, D]
        out = torch.zeros(num_nodes, self.num_heads, self.head_dim, device=x.device)
        out.scatter_add_(0, dst.unsqueeze(1).unsqueeze(2).expand_as(msg), msg)
        return out.view(num_nodes, -1)  # [N, H*D]


# ---------------------------------------------------------------------------
#  HAN (Heterogeneous Attention Network) 基线
# ---------------------------------------------------------------------------

class HANLayer(nn.Module):
    """异质注意力网络层 (Wang et al., 2019)

    节点级注意力 + 语义级注意力。
    对每种关系先做 GAT 聚合，再用 semantic attention 融合。
    """
    def __init__(self, in_dim, out_dim, num_relations):
        super().__init__()
        self.num_relations = num_relations
        self.gat_layers = nn.ModuleList([
            GATLayer(in_dim, out_dim, num_heads=1) for _ in range(num_relations)
        ])
        # 语义级注意力
        self.semantic_q = nn.Linear(out_dim, 1, bias=False)

    def forward(self, x, edge_index, edge_type, num_nodes, edge_weight=None):
        z_list = []
        for r in range(self.num_relations):
            mask = (edge_type == r)
            if mask.sum() == 0:
                z_list.append(torch.zeros(num_nodes, self.gat_layers[0].num_heads * self.gat_layers[0].head_dim, device=x.device))
                continue
            ei_r = edge_index[:, mask]
            ew_r = edge_weight[mask] if edge_weight is not None else None
            z_r = self.gat_layers[r](x, ei_r, num_nodes, ew_r)
            z_list.append(z_r)
        Z = torch.stack(z_list, dim=0)  # [R, N, D]
        # 语义注意力权重
        w = self.semantic_q(Z).squeeze(-1)  # [R, N]
        beta = torch.softmax(w, dim=0).unsqueeze(-1)  # [R, N, 1]
        out = (beta * Z).sum(dim=0)  # [N, D]
        return out


class HANModel(nn.Module):
    """HAN 两层模型"""
    def __init__(self, in_dim, hidden_dim, out_dim, num_relations):
        super().__init__()
        self.han1 = HANLayer(in_dim, hidden_dim, num_relations)
        self.han2 = HANLayer(hidden_dim, hidden_dim, num_relations)
        self.out = nn.Linear(hidden_dim, out_dim)

    def forward(self, x, edge_index, edge_type, edge_weight=None):
        num_nodes = x.size(0)
        h = self.han1(x, edge_index, edge_type, num_nodes, edge_weight)
        h = F.relu(h)
        h = self.han2(h, edge_index, edge_type, num_nodes, edge_weight)
        h = F.relu(h)
        return self.out(h)


# ---------------------------------------------------------------------------
#  DGI (Deep Graph Infomax) 自监督基线
# ---------------------------------------------------------------------------

class GCNEncoder(nn.Module):
    """GCN 编码器，用于 DGI"""
    def __init__(self, in_dim, hidden_dim):
        super().__init__()
        self.gcn1 = GCNLayer(in_dim, hidden_dim)
        self.gcn2 = GCNLayer(hidden_dim, hidden_dim)

    def forward(self, x, edge_index, num_nodes, edge_weight=None):
        h = F.relu(self.gcn1(x, edge_index, num_nodes, edge_weight))
        h = self.gcn2(h, edge_index, num_nodes, edge_weight)
        return h


class DGI(nn.Module):
    """Deep Graph Infomax (Velickovic et al., 2019)

    自监督对比学习：最大化节点表示与图级摘要之间的互信息。
    训练完成后冻结编码器，在下游加线性分类头。
    """
    def __init__(self, in_dim, hidden_dim):
        super().__init__()
        self.encoder = GCNEncoder(in_dim, hidden_dim)
        self.discriminator = nn.Bilinear(hidden_dim, hidden_dim, 1)

    def forward(self, x, edge_index, num_nodes, edge_weight=None):
        # 正样本：原始图
        h_pos = self.encoder(x, edge_index, num_nodes, edge_weight)
        summary = torch.sigmoid(h_pos.mean(dim=0, keepdim=True))  # [1, D]
        # 负样本：打乱节点特征
        perm = torch.randperm(num_nodes, device=x.device)
        h_neg = self.encoder(x[perm], edge_index, num_nodes, edge_weight)
        # 判别
        pos_score = self.discriminator(h_pos, summary.expand_as(h_pos)).squeeze(-1)
        neg_score = self.discriminator(h_neg, summary.expand_as(h_neg)).squeeze(-1)
        return pos_score, neg_score, h_pos

    def encode(self, x, edge_index, num_nodes, edge_weight=None):
        return self.encoder(x, edge_index, num_nodes, edge_weight)


class DGIClassifier(nn.Module):
    """DGI + 线性分类头"""
    def __init__(self, in_dim, hidden_dim, out_dim):
        super().__init__()
        self.dgi = DGI(in_dim, hidden_dim)
        self.classifier = nn.Linear(hidden_dim, out_dim)
        self.hidden_dim = hidden_dim

    def forward(self, x, edge_index, edge_type, edge_weight=None):
        num_nodes = x.size(0)
        h = self.dgi.encode(x, edge_index, num_nodes, edge_weight)
        return self.classifier(h)

    def pretrain_step(self, x, edge_index, num_nodes, edge_weight=None):
        """DGI 自监督预训练一步"""
        pos_score, neg_score, _ = self.dgi(x, edge_index, num_nodes, edge_weight)
        loss = -torch.mean(F.logsigmoid(pos_score) + F.logsigmoid(-neg_score))
        return loss


# ---------------------------------------------------------------------------
#  MR_GNN 主模型
# ---------------------------------------------------------------------------

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
        else:  # rgcn_group = GE-MRGNN
            self.rgcn1 = RGCNLayer(in_dim, hidden_dim, num_relations)
            self.group1 = GroupEnhance(hidden_dim)
            self.rgcn2 = RGCNLayer(hidden_dim, hidden_dim, num_relations)
            self.group2 = GroupEnhance(hidden_dim)
            self.out = nn.Linear(hidden_dim, out_dim)

    def forward(self, x, edge_index, edge_type, edge_weight=None):
        num_nodes = x.size(0)
        if self.model_type == 'gcn':
            h = F.relu(self.gcn1(x, edge_index, num_nodes, edge_weight))
            h = F.relu(self.gcn2(h, edge_index, num_nodes, edge_weight))
            return self.out(h)
        elif self.model_type == 'gat':
            h = F.relu(self.gat1(x, edge_index, num_nodes, edge_weight))
            h = F.relu(self.gat2(h, edge_index, num_nodes, edge_weight))
            return self.out(h)
        elif self.model_type == 'rgcn':
            h = F.relu(self.rgcn1(x, edge_index, edge_type, num_nodes, edge_weight))
            h = F.relu(self.rgcn2(h, edge_index, edge_type, num_nodes, edge_weight))
            return self.out(h)
        else:
            h = F.relu(self.rgcn1(x, edge_index, edge_type, num_nodes, edge_weight))
            h = self.group1(h, edge_index, num_nodes, edge_weight)
            h = F.relu(self.rgcn2(h, edge_index, edge_type, num_nodes, edge_weight))
            h = self.group2(h, edge_index, num_nodes, edge_weight)
            return self.out(h)


def split_dataset(labels, train_ratio=0.6, val_ratio=0.2, seed=42):
    """将有标签的节点划分为训练/验证/测试集"""
    num_nodes = labels.size(0)
    labeled_idx = (labels >= 0).nonzero(as_tuple=True)[0]
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
    edge_weight = None
    if (base / 'edge_weight.pt').exists():
        edge_weight = torch.load(base / 'edge_weight.pt')
    labels_bot = None
    labels_stance = None
    if (base / 'labels_bot.pt').exists():
        labels_bot = torch.load(base / 'labels_bot.pt')
    if (base / 'labels_stance.pt').exists():
        labels_stance = torch.load(base / 'labels_stance.pt')
    return features, edge_index, edge_type, edge_weight, labels_bot, labels_stance


# ---------------------------------------------------------------------------
#  指标计算
# ---------------------------------------------------------------------------

def compute_metrics_sklearn(y_true, y_prob, y_pred):
    """使用 sklearn 计算全套指标（更稳健、可复现）"""
    y_t = y_true.cpu().numpy()
    y_p = y_prob.cpu().numpy()
    y_d = y_pred.cpu().numpy()
    metrics = {}
    metrics['accuracy'] = float((y_d == y_t).mean())
    metrics['precision'] = precision_score(y_t, y_d, zero_division=0)
    metrics['recall'] = recall_score(y_t, y_d, zero_division=0)
    metrics['f1'] = f1_score(y_t, y_d, zero_division=0)
    try:
        metrics['roc_auc'] = roc_auc_score(y_t, y_p)
    except ValueError:
        metrics['roc_auc'] = 0.0
    try:
        metrics['pr_auc'] = average_precision_score(y_t, y_p)
    except ValueError:
        metrics['pr_auc'] = 0.0
    cm = confusion_matrix(y_t, y_d, labels=[0, 1])
    metrics['confusion_matrix'] = cm
    fpr_arr, tpr_arr, _ = roc_curve(y_t, y_p)
    prec_arr, rec_arr, _ = precision_recall_curve(y_t, y_p)
    metrics['roc_curve'] = (fpr_arr, tpr_arr)
    metrics['pr_curve'] = (prec_arr, rec_arr)
    return metrics


def compute_metrics_manual(y_true, y_prob, y_pred):
    """手写指标计算（sklearn 不可用时的 fallback）"""
    y_m = y_true.float()
    yhat_m = y_prob
    y_d = y_pred
    metrics = {}
    metrics['accuracy'] = (y_d == y_true).float().mean().item()
    tp = ((y_d == 1) & (y_true == 1)).sum().item()
    fp = ((y_d == 1) & (y_true == 0)).sum().item()
    fn = ((y_d == 0) & (y_true == 1)).sum().item()
    tn = ((y_d == 0) & (y_true == 0)).sum().item()
    metrics['precision'] = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    metrics['recall'] = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    p, r = metrics['precision'], metrics['recall']
    metrics['f1'] = 2 * p * r / (p + r) if (p + r) > 0 else 0.0
    metrics['confusion_matrix'] = np.array([[tn, fp], [fn, tp]])
    # 手写 ROC/PR 曲线
    th = torch.linspace(0, 1, 101)
    tpr_list, fpr_list, prec_list, rec_list = [], [], [], []
    for t in th:
        pred_t = (yhat_m >= t).long()
        tp_t = ((pred_t == 1) & (y_true == 1)).sum().item()
        fp_t = ((pred_t == 1) & (y_true == 0)).sum().item()
        fn_t = ((pred_t == 0) & (y_true == 1)).sum().item()
        tn_t = ((pred_t == 0) & (y_true == 0)).sum().item()
        tpr_list.append(tp_t / (tp_t + fn_t) if (tp_t + fn_t) > 0 else 0.0)
        fpr_list.append(fp_t / (fp_t + tn_t) if (fp_t + tn_t) > 0 else 0.0)
        rec_list.append(tp_t / (tp_t + fn_t) if (tp_t + fn_t) > 0 else 0.0)
        prec_list.append(tp_t / (tp_t + fp_t) if (tp_t + fp_t) > 0 else 1.0)
    fpr_arr = np.array(fpr_list)
    tpr_arr = np.array(tpr_list)
    prec_arr = np.array(prec_list)
    rec_arr = np.array(rec_list)
    roc_sorted = sorted(zip(fpr_arr, tpr_arr))
    auc = sum((roc_sorted[i][0] - roc_sorted[i-1][0]) * (roc_sorted[i][1] + roc_sorted[i-1][1]) * 0.5 for i in range(1, len(roc_sorted)))
    pr_sorted = sorted(zip(rec_arr, prec_arr))
    pr_auc = sum((pr_sorted[i][0] - pr_sorted[i-1][0]) * (pr_sorted[i][1] + pr_sorted[i-1][1]) * 0.5 for i in range(1, len(pr_sorted)))
    metrics['roc_auc'] = auc
    metrics['pr_auc'] = pr_auc
    metrics['roc_curve'] = (fpr_arr, tpr_arr)
    metrics['pr_curve'] = (prec_arr, rec_arr)
    return metrics


def compute_all_metrics(y_true, y_prob, y_pred):
    """统一入口：优先 sklearn，fallback 到手写"""
    if HAS_SKLEARN:
        return compute_metrics_sklearn(y_true, y_prob, y_pred)
    return compute_metrics_manual(y_true, y_prob, y_pred)


# ---------------------------------------------------------------------------
#  保存结果
# ---------------------------------------------------------------------------

def save_results(out_dir, model_type, train_acc, val_acc, metrics, model, hidden, num_rel, in_dim, out_dim):
    """保存模型、指标、曲线数据"""
    out_dir.mkdir(parents=True, exist_ok=True)

    # 保存模型
    model_path = out_dir / f'{model_type}_model.pt'
    torch.save({
        'model_state_dict': model.state_dict(),
        'model_type': model_type,
        'hidden_dim': hidden,
        'num_relations': num_rel,
        'in_dim': in_dim,
        'out_dim': out_dim,
        'test_metrics': {k: v.tolist() if isinstance(v, np.ndarray) else v
                         for k, v in metrics.items() if k not in ('roc_curve', 'pr_curve')},
    }, model_path)
    print(f"模型已保存到: {model_path}")

    # ROC 曲线 CSV
    fpr_arr, tpr_arr = metrics['roc_curve']
    roc_lines = ["fpr,tpr"]
    for f_val, t_val in zip(fpr_arr, tpr_arr):
        roc_lines.append(f"{float(f_val):.6f},{float(t_val):.6f}")
    (out_dir / 'roc.csv').write_text("\n".join(roc_lines), encoding='utf-8')

    # PR 曲线 CSV
    prec_arr, rec_arr = metrics['pr_curve']
    pr_lines = ["precision,recall"]
    for p_val, r_val in zip(prec_arr, rec_arr):
        pr_lines.append(f"{float(p_val):.6f},{float(r_val):.6f}")
    (out_dir / 'pr.csv').write_text("\n".join(pr_lines), encoding='utf-8')

    # 混淆矩阵
    cm = metrics['confusion_matrix']
    cm_lines = ["TN,FP,FN,TP"]
    cm_lines.append(f"{cm[0,0]},{cm[0,1]},{cm[1,0]},{cm[1,1]}")
    (out_dir / 'confusion_matrix.csv').write_text("\n".join(cm_lines), encoding='utf-8')

    # Summary
    summary = (
        f"model={model_type}\n"
        f"train_acc={train_acc:.6f}\n"
        f"val_acc={val_acc:.6f}\n"
        f"test_acc={metrics['accuracy']:.6f}\n"
        f"precision={metrics['precision']:.6f}\n"
        f"recall={metrics['recall']:.6f}\n"
        f"f1={metrics['f1']:.6f}\n"
        f"roc_auc={metrics['roc_auc']:.6f}\n"
        f"pr_auc={metrics['pr_auc']:.6f}\n"
    )
    (out_dir / 'summary.txt').write_text(summary, encoding='utf-8')
    print(summary)


# ---------------------------------------------------------------------------
#  主训练流程
# ---------------------------------------------------------------------------

def main():
    args = sys.argv

    # 解析命令行参数
    def get_arg(name, default=None, cast=str):
        if name in args:
            try:
                return cast(args[args.index(name) + 1])
            except Exception:
                return default
        return default

    dataset_dir = Path(get_arg('--dataset', str(Path(__file__).parent)))
    task = get_arg('--task', 'bot')
    epochs = get_arg('--epochs', 100, int)
    hidden = get_arg('--hidden', 64, int)
    lr = get_arg('--lr', 0.001, float)
    patience = get_arg('--patience', 20, int)
    max_edges = get_arg('--max_edges', None, int)
    model_type = get_arg('--model', 'rgcn_group')
    show_top = get_arg('--show_top', 0, int)
    out_dir = get_arg('--out_dir', None)
    if out_dir is not None:
        out_dir = Path(out_dir)
    seed = get_arg('--seed', None, int)
    train_ratio = get_arg('--train_ratio', 0.6, float)
    val_ratio = get_arg('--val_ratio', 0.2, float)
    use_amp = '--amp' in args
    dgi_pretrain_epochs = get_arg('--dgi_epochs', 50, int)

    if seed is not None:
        set_seed(seed)

    # 设备选择
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"使用设备: {device}")

    # 加载数据
    features, edge_index, edge_type, edge_weight, labels_bot, labels_stance = load_dataset(dataset_dir)

    num_nodes = features.size(0)
    if max_edges is not None and edge_index.size(1) > max_edges:
        idx = torch.arange(edge_index.size(1))[:max_edges]
        edge_index = edge_index[:, idx]
        edge_type = edge_type[idx]
        if edge_weight is not None:
            edge_weight = edge_weight[idx]

    num_rel = int(edge_type.max().item()) + 1 if edge_type.numel() > 0 else 1

    # 标签与损失
    if task == 'bot' and labels_bot is not None:
        labels = labels_bot
        out_dim = 1
        pos_weight_val = get_arg('--pos_weight', None, float)
        if pos_weight_val is None:
            pos = (labels == 1).sum().item()
            neg = (labels == 0).sum().item()
            pos_weight_val = (neg / max(1, pos)) if (pos + neg) > 0 else 1.0
        criterion = nn.BCEWithLogitsLoss(pos_weight=torch.tensor(pos_weight_val, device=device))
        is_multiclass = False
    elif task == 'stance' and labels_stance is not None:
        labels = labels_stance
        out_dim = int(labels.max().item()) + 1
        criterion = nn.CrossEntropyLoss()
        is_multiclass = True
    else:
        print("未找到有效标签文件")
        return

    # 划分数据集（train_ratio / val_ratio 从 CLI 传入）
    train_mask, val_mask, test_mask = split_dataset(
        labels, train_ratio=train_ratio, val_ratio=val_ratio,
        seed=seed if seed else 42
    )
    print(f"数据集划分: 训练={train_mask.sum().item()}, 验证={val_mask.sum().item()}, 测试={test_mask.sum().item()}")

    # edge_weight 归一化
    if edge_weight is not None:
        ew_min = edge_weight.min()
        ew_max = edge_weight.max()
        if ew_max - ew_min > 1e-8:
            edge_weight = (edge_weight - ew_min) / (ew_max - ew_min) + 0.1
        else:
            edge_weight = torch.ones_like(edge_weight)

    # 移至设备
    features = features.to(device)
    edge_index = edge_index.to(device)
    edge_type = edge_type.to(device)
    if edge_weight is not None:
        edge_weight = edge_weight.to(device)
    labels = labels.to(device)
    train_mask = train_mask.to(device)
    val_mask = val_mask.to(device)
    test_mask = test_mask.to(device)

    if model_type == 'gcn':
        edge_type = torch.zeros_like(edge_type)
        num_rel = 1

    # LPA 特殊处理
    if model_type == 'lpa':
        adj = [[] for _ in range(num_nodes)]
        for i in range(edge_index.size(1)):
            s = int(edge_index[0, i])
            d = int(edge_index[1, i])
            adj[d].append(s)
        pred = labels.clone().cpu()
        for _ in range(5):
            for u in range(num_nodes):
                if pred[u] < 0 and len(adj[u]) > 0:
                    vs = torch.tensor([pred[v].item() for v in adj[u] if pred[v] >= 0], dtype=torch.long)
                    if vs.numel() > 0:
                        val = int((vs.float().mean() > 0.5).item()) if not is_multiclass else int(torch.mode(vs).values.item())
                        pred[u] = val
        mask_cpu = (labels.cpu() >= 0)
        acc = (pred[mask_cpu] == labels.cpu()[mask_cpu]).float().mean().item()
        print(f'LPA acc={acc:.4f}')
        return

    # 构建模型
    if model_type == 'han':
        model = HANModel(features.size(1), hidden, out_dim, num_rel).to(device)
    elif model_type == 'dgi':
        model = DGIClassifier(features.size(1), hidden, out_dim).to(device)
    else:
        model = MR_GNN(features.size(1), hidden, out_dim, num_rel, model_type=model_type).to(device)

    # DGI 自监督预训练
    if model_type == 'dgi':
        print(f"\n[DGI] 自监督预训练 {dgi_pretrain_epochs} epochs ...")
        pre_opt = torch.optim.Adam(model.dgi.parameters(), lr=lr)
        for ep in range(dgi_pretrain_epochs):
            model.train()
            pre_loss = model.pretrain_step(features, edge_index, num_nodes, edge_weight)
            pre_opt.zero_grad()
            pre_loss.backward()
            pre_opt.step()
            if (ep + 1) % 10 == 0:
                print(f"  DGI Pretrain Epoch {ep+1}/{dgi_pretrain_epochs} | Loss: {pre_loss.item():.4f}")
        # 冻结编码器
        for p in model.dgi.parameters():
            p.requires_grad = False
        print("[DGI] 预训练完成，冻结编码器，开始下游训练\n")

    optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=lr, weight_decay=1e-5)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=10)
    scaler = torch.amp.GradScaler('cuda') if (use_amp and device.type == 'cuda') else None

    # Early stopping
    best_val_loss = float('inf')
    best_model_state = None
    patience_counter = 0

    model_names = {
        'rgcn_group': 'GE-MRGNN (群组增强的多关系图神经网络)',
        'rgcn': 'RGCN (多关系图卷积网络)',
        'gcn': 'GCN (图卷积网络)',
        'gat': 'GAT (图注意力网络)',
        'han': 'HAN (异质注意力网络)',
        'dgi': 'DGI (深度图信息最大化)',
        'lpa': 'LPA (标签传播算法)'
    }
    display_name = model_names.get(model_type, model_type.upper())
    print(f"\n开始训练 {display_name}...")
    print(f"参数: hidden={hidden}, lr={lr}, epochs={epochs}, patience={patience}, "
          f"train_ratio={train_ratio}, val_ratio={val_ratio}, AMP={use_amp}")
    print("-" * 60)

    for ep in range(epochs):
        # === 训练 ===
        model.train()
        if use_amp and scaler is not None:
            with torch.amp.autocast('cuda'):
                logits = model(features, edge_index, edge_type, edge_weight)
                if is_multiclass:
                    loss = criterion(logits[train_mask], labels[train_mask])
                else:
                    loss = criterion(logits[train_mask].squeeze(1), labels[train_mask].float())
            optimizer.zero_grad()
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            logits = model(features, edge_index, edge_type, edge_weight)
            if is_multiclass:
                loss = criterion(logits[train_mask], labels[train_mask])
            else:
                loss = criterion(logits[train_mask].squeeze(1), labels[train_mask].float())
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        with torch.no_grad():
            if is_multiclass:
                pred = logits.argmax(dim=1)
            else:
                pred = (logits.sigmoid().squeeze(1) > 0.5).long()
            train_acc = (pred[train_mask] == labels[train_mask]).float().mean().item()

        # === 验证 ===
        model.eval()
        with torch.no_grad():
            logits = model(features, edge_index, edge_type, edge_weight)
            if is_multiclass:
                val_loss = criterion(logits[val_mask], labels[val_mask]).item()
                val_pred = logits.argmax(dim=1)
                val_acc = (val_pred[val_mask] == labels[val_mask]).float().mean().item()
            else:
                val_loss = criterion(logits[val_mask].squeeze(1), labels[val_mask].float()).item()
                val_pred = (logits.sigmoid().squeeze(1) > 0.5).long()
                val_acc = (val_pred[val_mask] == labels[val_mask]).float().mean().item()

        scheduler.step(val_loss)

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_model_state = {k: v.clone() for k, v in model.state_dict().items()}
            patience_counter = 0
        else:
            patience_counter += 1

        if (ep + 1) % 10 == 0 or ep == 0:
            print(f'Epoch {ep+1:3d}/{epochs} | Train Loss: {loss.item():.4f} | '
                  f'Train Acc: {train_acc:.4f} | Val Loss: {val_loss:.4f} | Val Acc: {val_acc:.4f}')

        if patience_counter >= patience:
            print(f'\nEarly stopping at epoch {ep+1}')
            break

    # 加载最佳模型
    if best_model_state is not None:
        model.load_state_dict(best_model_state)

    # === 测试 ===
    model.eval()
    with torch.no_grad():
        logits = model(features, edge_index, edge_type, edge_weight)
        if is_multiclass:
            test_pred = logits.argmax(dim=1)
            test_acc = (test_pred[test_mask] == labels[test_mask]).float().mean().item()
            print("-" * 60)
            print(f"测试集准确率: {test_acc:.4f}")
        else:
            probs = logits.sigmoid().squeeze(1)
            test_pred = (probs > 0.5).long()
            # 全套指标
            metrics = compute_all_metrics(
                labels[test_mask].cpu(), probs[test_mask].cpu(), test_pred[test_mask].cpu()
            )
            print("-" * 60)
            print(f"测试集指标:")
            print(f"  Accuracy:  {metrics['accuracy']:.4f}")
            print(f"  Precision: {metrics['precision']:.4f}")
            print(f"  Recall:    {metrics['recall']:.4f}")
            print(f"  F1:        {metrics['f1']:.4f}")
            print(f"  ROC-AUC:   {metrics['roc_auc']:.4f}")
            print(f"  PR-AUC:    {metrics['pr_auc']:.4f}")
            cm = metrics['confusion_matrix']
            print(f"  混淆矩阵:  TN={cm[0,0]} FP={cm[0,1]} FN={cm[1,0]} TP={cm[1,1]}")
            print("-" * 60)

    if show_top > 0:
        with torch.no_grad():
            logits = model(features, edge_index, edge_type, edge_weight)
            if is_multiclass:
                probs = torch.softmax(logits, dim=1)
                conf, pred_cls = probs.max(dim=1)
                order = torch.argsort(conf, descending=True)[:show_top]
                print("Top预测示例:")
                for i in order.tolist():
                    print(f'  节点 {i} 预测类={pred_cls[i].item()} 置信度={conf[i].item():.4f} 标签={labels[i].item()}')
            else:
                probs = torch.sigmoid(logits.squeeze(1))
                order = torch.argsort(probs, descending=True)[:show_top]
                print("Top预测示例:")
                for i in order.tolist():
                    print(f'  节点 {i} 恶意概率={probs[i].item():.4f} 标签={labels[i].item()}')

    if out_dir is not None and not is_multiclass:
        save_results(out_dir, model_type, train_acc, val_acc, metrics, model, hidden, num_rel,
                     features.size(1), out_dim)
    elif out_dir is not None and is_multiclass:
        out_dir.mkdir(parents=True, exist_ok=True)
        torch.save({
            'model_state_dict': model.state_dict(),
            'model_type': model_type,
            'hidden_dim': hidden,
            'num_relations': num_rel,
            'in_dim': features.size(1),
            'out_dim': out_dim,
            'test_acc': test_acc,
        }, out_dir / f'{model_type}_model.pt')
        (out_dir / 'summary.txt').write_text(
            f'model={model_type}\ntrain_acc={train_acc:.6f}\nval_acc={val_acc:.6f}\ntest_acc={test_acc:.6f}\n',
            encoding='utf-8'
        )


if __name__ == '__main__':
    main()
