"""消融实验脚本

验证GE-MRGNN各模块的贡献：
1. 去除群组增强模块 (RGCN vs GE-MRGNN)
2. 去除多关系建模 (GCN vs GE-MRGNN)  
3. 不同隐层维度 (32/64/128)
4. 不同训练比例 (40%/60%/80%)
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
    
    # 读取结果
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
    # 消融1: 模块消融 (已在对比实验中完成)
    # ============================================
    print("\n【消融实验1: 模块消融】")
    print("- GE-MRGNN (完整模型)")
    print("- RGCN (去除群组增强)")
    print("- GCN (去除多关系建模)")
    print("→ 已在对比实验中完成，结果见 results/ 目录")
    
    # ============================================
    # 消融2: 隐层维度消融
    # ============================================
    print("\n【消融实验2: 隐层维度消融】")
    
    for hidden in [32, 64, 128]:
        name = f"hidden_{hidden}"
        result = run_experiment(
            name=name,
            model='rgcn_group',
            hidden=hidden,
            epochs=50
        )
        if result:
            results.append((f"Hidden={hidden}", result))
    
    # ============================================
    # 消融3: 训练比例消融 (需要修改代码支持)
    # ============================================
    print("\n【消融实验3: 训练比例消融】")
    print("→ 需要修改数据划分比例，暂时跳过")
    
    # ============================================
    # 汇总结果
    # ============================================
    print("\n" + "="*60)
    print("消融实验结果汇总")
    print("="*60)
    
    # 读取之前的对比实验结果
    base_dir = Path(__file__).parent / 'results'
    
    print("\n【模块消融结果】")
    for model_name, display in [('ge-mrgcn', 'GE-MRGNN (完整)'), 
                                 ('rgcn_group', 'GE-MRGNN (完整)'),
                                 ('rgcn', 'RGCN (无群组增强)'), 
                                 ('gcn', 'GCN (无多关系)')]:
        summary = base_dir / model_name / 'summary.txt'
        if summary.exists():
            content = summary.read_text(encoding='utf-8')
            for line in content.strip().split('\n'):
                if 'test_acc' in line:
                    acc = line.split('=')[1]
                    print(f"  {display}: {float(acc)*100:.2f}%")
                    break
    
    print("\n【隐层维度消融结果】")
    ablation_dir = Path(__file__).parent / 'ablation_results'
    for hidden in [32, 64, 128]:
        summary = ablation_dir / f'hidden_{hidden}' / 'summary.txt'
        if summary.exists():
            content = summary.read_text(encoding='utf-8')
            for line in content.strip().split('\n'):
                if 'test_acc' in line:
                    acc = line.split('=')[1]
                    print(f"  Hidden={hidden}: {float(acc)*100:.2f}%")
                    break
    
    print("\n" + "="*60)
    print("消融实验完成!")
    print("="*60)


if __name__ == '__main__':
    main()
