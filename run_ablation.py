"""消融实验脚本

验证GE-MRGNN各模块的贡献：
1. 模块消融: 去除群组增强 (RGCN) / 去除多关系 (GCN)
2. 隐层维度消融 (32/64/128)
3. 训练比例消融 (40%/60%/80%)
4. 基线对比: HAN, DGI
"""

import subprocess
import sys
from pathlib import Path


def run_experiment(name, model, hidden=64, epochs=50, seed=42, extra_args=None):
    """运行单个实验"""
    print(f"\n{'='*60}")
    print(f"消融实验: {name}")
    print(f"{'='*60}")

    out_dir = Path(__file__).parent / 'ablation_results' / name.replace(' ', '_').replace('/', '_')

    cmd = [
        sys.executable, 'train_mrgnn.py',
        '--model', model,
        '--hidden', str(hidden),
        '--epochs', str(epochs),
        '--seed', str(seed),
        '--out_dir', str(out_dir)
    ]

    if extra_args:
        cmd.extend(extra_args)

    subprocess.run(cmd, cwd=Path(__file__).parent)

    summary_file = out_dir / 'summary.txt'
    if summary_file.exists():
        return summary_file.read_text(encoding='utf-8')
    return None


def main():
    results = []

    print("\n" + "="*60)
    print("GE-MRGNN 消融实验")
    print("="*60)

    # ============================================
    # 消融1: 模块消融
    # ============================================
    print("\n【消融实验1: 模块消融】")
    for model, label in [('rgcn_group', 'GE-MRGNN_完整'),
                          ('rgcn', 'RGCN_无群组增强'),
                          ('gcn', 'GCN_无多关系')]:
        result = run_experiment(name=f"module_{label}", model=model, epochs=50)
        if result:
            results.append((label, result))

    # ============================================
    # 消融2: 隐层维度消融
    # ============================================
    print("\n【消融实验2: 隐层维度消融】")
    for hidden in [32, 64, 128]:
        name = f"hidden_{hidden}"
        result = run_experiment(name=name, model='rgcn_group', hidden=hidden, epochs=50)
        if result:
            results.append((f"Hidden={hidden}", result))

    # ============================================
    # 消融3: 训练比例消融（通过 --train_ratio 传入）
    # ============================================
    print("\n【消融实验3: 训练比例消融】")
    for ratio in [0.4, 0.6, 0.8]:
        name = f"train_ratio_{int(ratio*100)}"
        result = run_experiment(
            name=name, model='rgcn_group', epochs=50,
            extra_args=['--train_ratio', str(ratio)]
        )
        if result:
            results.append((f"TrainRatio={int(ratio*100)}%", result))

    # ============================================
    # 消融4: 基线模型对比 (HAN, DGI)
    # ============================================
    print("\n【消融实验4: 基线模型对比】")
    for model, label in [('han', 'HAN'), ('dgi', 'DGI')]:
        result = run_experiment(name=f"baseline_{label}", model=model, epochs=50)
        if result:
            results.append((label, result))

    # ============================================
    # 汇总
    # ============================================
    print("\n" + "="*60)
    print("消融实验结果汇总")
    print("="*60)

    ablation_dir = Path(__file__).parent / 'ablation_results'
    for sub in sorted(ablation_dir.iterdir()):
        summary = sub / 'summary.txt'
        if summary.exists():
            content = summary.read_text(encoding='utf-8').strip()
            print(f"\n  [{sub.name}]")
            for line in content.split('\n'):
                print(f"    {line}")

    print("\n" + "="*60)
    print("消融实验完成!")
    print("="*60)


if __name__ == '__main__':
    main()
