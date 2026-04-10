"""GE-MRGNN 网络安全团伙检测系统

本项目基于多关系图神经网络(R-GCN)和群组增强模块，
用于检测网络安全领域的团伙型协同攻击行为。

模型架构:
- 多关系安全图编码模块 (R-GCN): 为不同类型的网络连接学习独立的变换矩阵
- 群组增强模块 (GroupEnhance): 通过邻居聚合增强节点的群组感知能力  
- 团伙攻击识别模块: 基于节点表示进行恶意团伙分类

参考论文:
- R-GCN: Modeling Relational Data with Graph Convolutional Networks (Schlichtkrull et al., 2018)
- GraphSAGE: Inductive Representation Learning on Large Graphs (Hamilton et al., 2017)

使用方法:
    python main.py --help           # 查看帮助
    python main.py --run_demo       # 运行演示
    python main.py --run_comparison # 运行对比实验
"""

import sys
import subprocess
from pathlib import Path


def print_help():
    """打印帮助信息"""
    print("""
GE-MRGNN 网络安全团伙检测系统
========================================

使用方法:
    python main.py [选项]

选项:
    --help              显示此帮助信息
    --run_demo          运行单个模型演示
    --run_comparison    运行对比实验 (全部模型)
    --run_ablation      运行消融实验
    --model MODEL       模型类型: gcn, gat, rgcn, rgcn_group, han, dgi, lpa (默认: rgcn_group)
    --epochs N          训练轮数 (默认: 100)
    --hidden N          隐藏层维度 (默认: 64)
    --seed N            随机种子 (默认: 42)
    --train_ratio F     训练集比例 (默认: 0.6)
    --val_ratio F       验证集比例 (默认: 0.2)
    --amp               启用混合精度训练 (需 CUDA)

示例:
    python main.py --run_demo
    python main.py --run_demo --model han --epochs 50
    python main.py --run_comparison
    python main.py --run_ablation
""")


def run_single_model(model_type='rgcn_group', epochs=100, hidden=64, seed=42):
    """运行单个模型训练"""
    model_names = {
        'rgcn_group': 'GE-MRGNN',
        'rgcn': 'RGCN',
        'gcn': 'GCN',
        'gat': 'GAT',
        'lpa': 'LPA'
    }
    display_name = model_names.get(model_type, model_type.upper())
    
    print(f"\n{'='*60}")
    print(f"训练模型: {display_name}")
    print(f"{'='*60}")
    
    cmd = [
        sys.executable, 'train_mrgnn.py',
        '--model', model_type,
        '--epochs', str(epochs),
        '--hidden', str(hidden),
        '--seed', str(seed),
        '--out_dir', f'results/{model_type}'
    ]
    subprocess.run(cmd, cwd=Path(__file__).parent)


def run_comparison_experiments(epochs=100, hidden=64, seed=42):
    """运行对比实验"""
    print("\n" + "="*60)
    print("开始对比实验")
    print("="*60)
    print(f"参数: epochs={epochs}, hidden={hidden}, seed={seed}")
    
    models = [
        ('gcn', 'GCN (单关系图卷积网络)'),
        ('gat', 'GAT (图注意力网络)'),
        ('rgcn', 'RGCN (多关系图卷积网络)'),
        ('han', 'HAN (异质注意力网络)'),
        ('dgi', 'DGI (深度图信息最大化)'),
        ('rgcn_group', 'GE-MRGNN (群组增强的多关系图神经网络)'),
        ('lpa', 'LPA (标签传播算法)')
    ]
    
    results = []
    for model_type, model_name in models:
        print(f"\n{'='*60}")
        print(f"[{len(results)+1}/{len(models)}] {model_name}")
        print(f"{'='*60}")
        
        cmd = [
            sys.executable, 'train_mrgnn.py',
            '--model', model_type,
            '--epochs', str(epochs),
            '--hidden', str(hidden),
            '--seed', str(seed),
            '--out_dir', f'results/{model_type}'
        ]
        subprocess.run(cmd, cwd=Path(__file__).parent)
        results.append((model_type, model_name))
    
    # 汇总结果
    print("\n" + "="*60)
    print("实验结果汇总")
    print("="*60)
    
    results_dir = Path(__file__).parent / 'results'
    for model_type, model_name in results:
        summary_file = results_dir / model_type / 'summary.txt'
        if summary_file.exists():
            content = summary_file.read_text(encoding='utf-8')
            print(f"\n{model_name}:")
            for line in content.strip().split('\n'):
                print(f"  {line}")
        else:
            print(f"\n{model_name}: 结果文件未生成")


def main():
    args = sys.argv
    
    if '--help' in args or len(args) == 1:
        print_help()
        return
    
    # 解析参数
    epochs = 100
    if '--epochs' in args:
        epochs = int(args[args.index('--epochs') + 1])
    
    hidden = 64
    if '--hidden' in args:
        hidden = int(args[args.index('--hidden') + 1])
    
    seed = 42
    if '--seed' in args:
        seed = int(args[args.index('--seed') + 1])
    
    model_type = 'rgcn_group'
    if '--model' in args:
        model_type = args[args.index('--model') + 1]
    
    if '--run_comparison' in args:
        run_comparison_experiments(epochs=epochs, hidden=hidden, seed=seed)
    elif '--run_ablation' in args:
        subprocess.run([sys.executable, 'run_ablation.py'], cwd=Path(__file__).parent)
    elif '--run_demo' in args:
        run_single_model(model_type=model_type, epochs=epochs, hidden=hidden, seed=seed)
    else:
        print_help()


if __name__ == '__main__':
    main()
