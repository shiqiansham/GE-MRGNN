import torch
from pathlib import Path
import sys

if '--dataset' in sys.argv:
    i = sys.argv.index('--dataset')
    if i + 1 < len(sys.argv):
        base_dir = Path(sys.argv[i + 1])
    else:
        base_dir = Path(__file__).parent
else:
    base_dir = Path(__file__).parent

edge_index = torch.load(base_dir / 'edge_index.pt')
print("边的形状:", edge_index.shape)
print("前10条边:\n", edge_index[:, :10])

node_features = torch.load(base_dir / 'features.pt')
print("节点特征形状:", node_features.shape)

labels_stance_path = base_dir / 'labels_stance.pt'
if labels_stance_path.exists():
    labels_stance = torch.load(labels_stance_path)
    print("标签文件:", labels_stance_path.name)
    print("标签形状:", labels_stance.shape)
    unique, counts = torch.unique(labels_stance, return_counts=True)
    print("标签分布:", dict(zip(unique.tolist(), counts.tolist())))
else:
    print("没有找到 labels_stance.pt")

labels_bot_path = base_dir / 'labels_bot.pt'
if labels_bot_path.exists():
    labels_bot = torch.load(labels_bot_path)
    print("标签文件:", labels_bot_path.name)
    print("标签形状:", labels_bot.shape)
    unique, counts = torch.unique(labels_bot, return_counts=True)
    print("标签分布:", dict(zip(unique.tolist(), counts.tolist())))
else:
    print("没有找到 labels_bot.pt")

train_mask_path = base_dir / 'train_mask.pt'
if train_mask_path.exists():
    train_mask = torch.load(train_mask_path)
    print("训练集节点数:", train_mask.sum().item())
else:
    print("没有找到 train_mask.pt")

edge_type_path = base_dir / 'edge_type.pt'
if edge_type_path.exists():
    edge_type = torch.load(edge_type_path)
    print("边类型形状:", edge_type.shape)

edge_weight_path = base_dir / 'edge_weight.pt'
if edge_weight_path.exists():
    edge_weight = torch.load(edge_weight_path)
    print("边权重形状:", edge_weight.shape)
