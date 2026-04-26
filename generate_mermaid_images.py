import base64
import json
import urllib.parse
from pathlib import Path
import requests

def generate_mermaid_image(mermaid_text, output_path):
    # State format for mermaid.ink
    state = {
        "code": mermaid_text,
        "mermaid": '{"theme": "default"}',
        "autoSync": True,
        "updateDiagram": True
    }
    
    # Compress and encode
    import zlib
    json_str = json.dumps(state)
    # The new mermaid.ink uses base64 encoded pako deflated json or just base64 of the json
    # Let's try simple base64 of json first. Actually mermaid live editor uses base64url of the state json.
    b64_str = base64.urlsafe_b64encode(json_str.encode('utf-8')).decode('utf-8')
    url = f"https://mermaid.ink/img/pako:{b64_str}"
    
    # Try another way: just encode the raw code string as base64 and use standard url
    b64_code = base64.b64encode(mermaid_text.encode('utf-8')).decode('utf-8')
    url2 = f"https://mermaid.ink/img/{b64_code}?bgColor=!white"
    
    print(f"Downloading from: {url2}")
    response = requests.get(url2)
    if response.status_code == 200:
        with open(output_path, 'wb') as f:
            f.write(response.content)
        print(f"Successfully generated {output_path}")
    else:
        print(f"Failed to generate {output_path}. Status code: {response.status_code}, Response: {response.text[:100]}")

base_dir = Path(r"F:\GranduateDesign\DataGet\images")

# Graph 1
g1 = """flowchart TD 
    subgraph "前端展示层" 
        A1["Vue.js 可视化大盘"] 
        A2["ECharts 团伙图谱动态展示"] 
        A3["多模型评估指标看板"] 
    end 

    subgraph "服务接口层" 
        B1["FastAPI 微服务"] 
        B2["RESTful API 路由分发"] 
        B3["模型驻留与状态管理"] 
    end 

    subgraph "核心算法层" 
        C1["GE-MRGNN 检测引擎"] 
        C2["多关系图卷积模块 R-GCN"] 
        C3["群组增强模块 GroupEnhance"] 
        C4["BCE 损失与自适应优化"] 
    end 

    subgraph "数据处理层" 
        D1["多源异构安全数据"] 
        D2["特征工程与映射"] 
        D3["PyTorch Tensor 转换"] 
    end 

    A1 --- B1 
    A2 --- B1 
    A3 --- B1 
    B2 --- C1 
    C1 --- D2 
    D2 --- D1
"""

# Graph 2
g2 = """graph LR 
    A[输入层: 788维节点特征 + 多关系邻接矩阵] --> B[第一层 R-GCN: 学习不同边类型特征] 
    B --> C[第一层 GroupEnhance: 聚合一阶群组特性] 
    C --> D[第二层 R-GCN: 深层复杂关系提取] 
    D --> E[第二层 GroupEnhance: 高阶团伙行为感知] 
    E --> F[输出层: 线性映射 + Sigmoid 激活] 
    F --> G[输出: 节点恶意概率预测]
"""

# Graph 3
g3 = """graph TD 
    Center((目标节点 v_i)) 
    N1((邻居 v_1)) -- "关系 r_1 (如: 转发)" --> Center 
    N2((邻居 v_2)) -- "关系 r_1 (如: 转发)" --> Center 
    N3((邻居 v_3)) -- "关系 r_2 (如: 关注)" --> Center 
    N4((邻居 v_4)) -- "关系 r_3 (如: 同源IP)" --> Center 

    subgraph "R-GCN 聚合计算" 
        W1[权重矩阵 W_{r_1}] 
        W2[权重矩阵 W_{r_2}] 
        W3[权重矩阵 W_{r_3}] 
        W0[自环权重 W_0] 
    end 

    N1 -.-> W1 
    N2 -.-> W1 
    N3 -.-> W2 
    N4 -.-> W3 
    Center -.-> W0 

    W1 --> Sum[特征求和与归一化] 
    W2 --> Sum 
    W3 --> Sum 
    W0 --> Sum 

    Sum --> ReLU[ReLU 激活] 
    ReLU --> Out((更新后节点 v_i))
"""

generate_mermaid_image(g1, base_dir / "fig1.png")
generate_mermaid_image(g2, base_dir / "fig2.png")
generate_mermaid_image(g3, base_dir / "fig3.png")