"""GE-MRGNN REST API 服务

提供 RESTful API 进行模型推理和结果查询。

启动方式:
    python api_server.py
    python api_server.py --port 8000 --model_path results/rgcn_group/rgcn_group_model.pt

API 端点:
    GET  /                  健康检查
    GET  /api/status        模型与数据状态
    POST /api/predict       上传 nodes/edges CSV 或 .pt 文件进行推理
    GET  /api/nodes         查询节点预测结果（支持分页、排序、过滤）
    GET  /api/gangs         查询团伙子图
    GET  /api/metrics       查询模型评估指标
"""
import sys
import json
import io
import csv
import tempfile
from pathlib import Path

import torch
import numpy as np

try:
    from fastapi import FastAPI, UploadFile, File, Query, HTTPException
    from fastapi.responses import JSONResponse, HTMLResponse
    from fastapi.staticfiles import StaticFiles
    from fastapi.middleware.cors import CORSMiddleware
    import uvicorn
    HAS_FASTAPI = True
except ImportError:
    HAS_FASTAPI = False


# ---------------------------------------------------------------------------
#  全局状态
# ---------------------------------------------------------------------------
app_state = {
    "model": None,
    "model_type": None,
    "features": None,
    "edge_index": None,
    "edge_type": None,
    "edge_weight": None,
    "ip_map": {},
    "probs": None,        # 缓存的推理结果
    "gangs": None,         # 缓存的团伙子图
    "loaded": False,
}


def init_model(model_path, dataset_dir):
    """初始化模型和数据"""
    from inference import load_model_and_data, predict, extract_gang_subgraphs

    model, features, edge_index, edge_type, edge_weight, ip_map = load_model_and_data(
        model_path, dataset_dir
    )
    app_state["model"] = model
    app_state["features"] = features
    app_state["edge_index"] = edge_index
    app_state["edge_type"] = edge_type
    app_state["edge_weight"] = edge_weight
    app_state["ip_map"] = ip_map
    app_state["loaded"] = True

    # 预计算推理结果
    probs = predict(model, features, edge_index, edge_type, edge_weight)
    app_state["probs"] = probs

    # 预提取团伙子图
    mal_nodes = (probs > 0.5).nonzero(as_tuple=True)[0].tolist()
    gangs = extract_gang_subgraphs(mal_nodes, edge_index, max_hops=2)
    app_state["gangs"] = gangs

    print(f"模型加载完成: {model_path}")
    print(f"  节点数: {features.size(0)}, 边数: {edge_index.size(1)}")
    print(f"  恶意节点: {len(mal_nodes)}, 团伙: {len(gangs)}")


# ---------------------------------------------------------------------------
#  FastAPI App
# ---------------------------------------------------------------------------

def create_app():
    app = FastAPI(title="GE-MRGNN 网络安全团伙检测 API", version="1.0.0")

    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    @app.get("/")
    def index():
        """根路由：返回前端仪表盘页面"""
        html_path = Path(__file__).parent / 'static' / 'index.html'
        if html_path.exists():
            return HTMLResponse(html_path.read_text(encoding='utf-8'))
        return {"status": "ok", "model_loaded": app_state["loaded"], "docs": "/docs"}

    @app.get("/api/health")
    def health():
        return {"status": "ok", "model_loaded": app_state["loaded"]}

    @app.get("/api/status")
    def status():
        if not app_state["loaded"]:
            return {"loaded": False}
        probs = app_state["probs"]
        return {
            "loaded": True,
            "num_nodes": int(app_state["features"].size(0)),
            "num_edges": int(app_state["edge_index"].size(1)),
            "num_malicious": int((probs > 0.5).sum().item()),
            "num_gangs": len(app_state["gangs"]) if app_state["gangs"] else 0,
        }

    @app.get("/api/nodes")
    def get_nodes(
        page: int = Query(1, ge=1),
        page_size: int = Query(50, ge=1, le=500),
        sort_by: str = Query("prob_desc"),
        min_prob: float = Query(0.0, ge=0.0, le=1.0),
        malicious_only: bool = Query(False),
    ):
        if not app_state["loaded"]:
            raise HTTPException(400, "模型未加载")
        probs = app_state["probs"]
        ip_map = app_state["ip_map"]
        num_nodes = probs.size(0)

        # 过滤
        indices = list(range(num_nodes))
        if malicious_only:
            indices = [i for i in indices if probs[i].item() > 0.5]
        if min_prob > 0:
            indices = [i for i in indices if probs[i].item() >= min_prob]

        # 排序
        if sort_by == "prob_desc":
            indices.sort(key=lambda i: probs[i].item(), reverse=True)
        elif sort_by == "prob_asc":
            indices.sort(key=lambda i: probs[i].item())

        total = len(indices)
        start = (page - 1) * page_size
        end = start + page_size
        page_indices = indices[start:end]

        nodes = []
        for i in page_indices:
            nodes.append({
                "node_id": i,
                "ip": ip_map.get(i, f"node_{i}"),
                "probability": round(probs[i].item(), 6),
                "is_malicious": bool(probs[i].item() > 0.5),
            })

        return {
            "total": total,
            "page": page,
            "page_size": page_size,
            "nodes": nodes,
        }

    @app.get("/api/gangs")
    def get_gangs(
        min_size: int = Query(2, ge=1),
        top_k: int = Query(20, ge=1, le=100),
    ):
        if not app_state["loaded"] or not app_state["gangs"]:
            raise HTTPException(400, "模型未加载或无团伙数据")
        gangs = app_state["gangs"]
        ip_map = app_state["ip_map"]
        probs = app_state["probs"]

        result = []
        for sg in gangs:
            if sg['size'] < min_size:
                continue
            result.append({
                "nodes": sg['nodes'],
                "edges": sg['edges'],
                "size": sg['size'],
                "num_malicious": sg['num_malicious'],
                "node_labels": {str(n): ip_map.get(n, f"node_{n}") for n in sg['nodes']},
                "node_probs": {str(n): round(probs[n].item(), 6) for n in sg['nodes']},
            })
            if len(result) >= top_k:
                break

        return {"total": len(result), "gangs": result}

    @app.get("/api/metrics")
    def get_metrics():
        """返回最新的训练评估指标"""
        results_dir = Path(__file__).parent / 'results'
        metrics = {}
        for sub in results_dir.iterdir():
            summary = sub / 'summary.txt'
            if summary.exists():
                content = summary.read_text(encoding='utf-8').strip()
                m = {}
                for line in content.split('\n'):
                    if '=' in line:
                        k, v = line.split('=', 1)
                        try:
                            m[k.strip()] = float(v.strip())
                        except ValueError:
                            m[k.strip()] = v.strip()
                metrics[sub.name] = m
        return {"metrics": metrics}

    @app.get("/api/curves")
    def get_curves():
        """返回各模型的 ROC/PR 曲线数据"""
        results_dir = Path(__file__).parent / 'results'
        curves = {}
        for sub in results_dir.iterdir():
            roc_file = sub / 'roc.csv'
            pr_file = sub / 'pr.csv'
            c = {}
            if roc_file.exists():
                lines = roc_file.read_text(encoding='utf-8').strip().split('\n')
                header = lines[0].lower().split(',')
                fpr_list, tpr_list = [], []
                for line in lines[1:]:
                    parts = line.split(',')
                    if len(parts) < 2:
                        continue
                    # 兼容两种格式: "threshold,fpr,tpr" 和 "fpr,tpr"
                    if len(header) >= 3 and 'threshold' in header[0]:
                        fpr_list.append(float(parts[1]))
                        tpr_list.append(float(parts[2]))
                    else:
                        fpr_list.append(float(parts[0]))
                        tpr_list.append(float(parts[1]))
                c['roc'] = {'fpr': fpr_list, 'tpr': tpr_list}
            if pr_file.exists():
                lines = pr_file.read_text(encoding='utf-8').strip().split('\n')
                header = lines[0].lower().split(',')
                prec_list, rec_list = [], []
                for line in lines[1:]:
                    parts = line.split(',')
                    if len(parts) < 2:
                        continue
                    # 兼容: "threshold,precision,recall" 和 "precision,recall"
                    if len(header) >= 3 and 'threshold' in header[0]:
                        prec_list.append(float(parts[1]))
                        rec_list.append(float(parts[2]))
                    else:
                        prec_list.append(float(parts[0]))
                        rec_list.append(float(parts[1]))
                c['pr'] = {'precision': prec_list, 'recall': rec_list}
            if c:
                curves[sub.name] = c
        return {"curves": curves}

    @app.post("/api/predict")
    async def predict_upload(
        files: list[UploadFile] = File(...),
        threshold: float = Query(0.5, ge=0.0, le=1.0),
    ):
        """上传数据进行推理

        上传 features.pt + edge_index.pt + edge_type.pt 三个文件，
        系统根据文件名自动识别。
        """
        if not app_state["loaded"]:
            raise HTTPException(400, "模型未加载")

        from inference import predict as do_predict, extract_gang_subgraphs

        model = app_state["model"]

        try:
            # 按文件名分类
            file_map = {}
            for f in files:
                name = (f.filename or '').lower()
                file_map[name] = f
                print(f"[predict] 收到文件: {f.filename} ({f.size or '?'} bytes)")

            # 查找 .pt 文件（按文件名关键词匹配）
            pt_feat = pt_ei = pt_et = None
            for name, f in file_map.items():
                if 'feature' in name:
                    pt_feat = f
                elif 'edge_index' in name:
                    pt_ei = f
                elif 'edge_type' in name:
                    pt_et = f

            if pt_feat and pt_ei and pt_et:
                # .pt 格式
                feat_bytes = await pt_feat.read()
                features = torch.load(io.BytesIO(feat_bytes), map_location='cpu',
                                      weights_only=False)
                ei_bytes = await pt_ei.read()
                edge_index = torch.load(io.BytesIO(ei_bytes), map_location='cpu',
                                        weights_only=False)
                et_bytes = await pt_et.read()
                edge_type = torch.load(io.BytesIO(et_bytes), map_location='cpu',
                                       weights_only=False)
                edge_weight = None
                print(f"[predict] PT解析成功: features={features.shape}, "
                      f"edges={edge_index.shape}, types={edge_type.shape}")
            else:
                received = [f.filename for f in files]
                raise HTTPException(
                    400,
                    f"需要同时上传 features.pt、edge_index.pt、edge_type.pt 三个文件。"
                    f"收到的文件: {received}"
                )

            probs = do_predict(model, features, edge_index, edge_type, edge_weight)
            mal_nodes = (probs > threshold).nonzero(as_tuple=True)[0].tolist()
            gangs = extract_gang_subgraphs(mal_nodes, edge_index, max_hops=2)

            order = torch.argsort(probs, descending=True)
            top_nodes = []
            for i in range(min(50, probs.size(0))):
                idx = int(order[i])
                top_nodes.append({
                    "node_id": idx,
                    "probability": round(probs[idx].item(), 6),
                    "is_malicious": bool(probs[idx].item() > threshold),
                })

            return {
                "num_nodes": int(features.size(0)),
                "num_edges": int(edge_index.size(1)),
                "threshold": threshold,
                "num_malicious": len(mal_nodes),
                "num_gangs": len(gangs),
                "top_nodes": top_nodes,
                "gangs": gangs[:10],
            }
        except HTTPException:
            raise
        except Exception as e:
            import traceback
            traceback.print_exc()
            raise HTTPException(500, f"推理失败: {str(e)}")

    return app


# ---------------------------------------------------------------------------
#  启动入口
# ---------------------------------------------------------------------------

def main():
    if not HAS_FASTAPI:
        print("错误: 需要安装 fastapi 和 uvicorn")
        print("  pip install fastapi uvicorn python-multipart")
        return

    args = sys.argv

    def get_arg(name, default=None, cast=str):
        if name in args:
            try:
                return cast(args[args.index(name) + 1])
            except Exception:
                return default
        return default

    model_path = get_arg('--model_path', 'results/ge-mrgcn/rgcn_group_model.pt')
    dataset_dir = get_arg('--dataset', str(Path(__file__).parent))
    port = get_arg('--port', 8000, int)
    host = get_arg('--host', '127.0.0.1')

    app = create_app()

    # 加载模型
    if Path(model_path).exists():
        init_model(model_path, dataset_dir)
    else:
        print(f"警告: 模型文件不存在 ({model_path})，API 启动但推理不可用")

    print(f"\nAPI 服务启动: http://{host}:{port}")
    print(f"API 文档:     http://{host}:{port}/docs")
    uvicorn.run(app, host=host, port=port)


if __name__ == '__main__':
    main()
