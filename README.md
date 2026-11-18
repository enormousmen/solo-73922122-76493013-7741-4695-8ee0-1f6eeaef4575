# Micro-batch Image Classifier

为视觉部署场景（多台相机）提供的**限时图片分类**服务：FastAPI + CPU 版 PyTorch，
单进程微批（micro-batch）推理。提交图片时给出从受理起算的截止毫秒数；
超时或被取消的帧不会产出结果。

## 快速开始

```bash
docker compose up --build
# 服务: http://localhost:8000   交互文档: http://localhost:8000/docs
```

本地开发（无需 Docker）：

```bash
python -m venv .venv && .venv/bin/pip install -r requirements-dev.txt
.venv/bin/uvicorn app.main:app --port 8000
```

## API

| 方法 | 路径 | 说明 |
|---|---|---|
| POST | `/predict` | 提交图片（multipart `file`）+ 表单 `deadline_ms`（从受理起算）。返回 `order_id`。满额返回 **429** |
| GET | `/orders/{order_id}` | 按单号查询状态与结果 |
| POST | `/orders/{order_id}/cancel` | 按单号取消；已完成返回 **409** |
| POST | `/calibration` | 用 8～32 个互异已完成单号及 0～9 真值拟合校准温度；重叠请求返回 **409** |
| GET | `/stats` | 容量/状态计数（演示与观测用） |
| GET | `/health` | 存活探针 |

示例：

```bash
curl -X POST http://localhost:8000/predict \
  -F "file=@samples/sample_gradient.png" -F "deadline_ms=5000"
# {"order_id":"7f6770829f36","status":"queued",...}

curl http://localhost:8000/orders/7f6770829f36
# {"status":"completed","temperature":1.0,"result":{"predicted_class":1,"scores":[...10 个分数...]},...}

curl -X POST http://localhost:8000/orders/7f6770829f36/cancel
```

单号状态机：`queued → processing → completed`；`queued/processing → cancelled|expired`。
`result.scores` 为 10 类 softmax 分数，与提交单号一一对应。

## 温度校准（POST /calibration）

人工核验一批已完成单号的真值后，用该接口拟合单一温度 `T`，对在线分类分数做
置信度校准。请求体：

```json
{
  "samples": [
    {"order_id": "7f6770829f36", "label": 3},
    {"order_id": "a1b2c3d4e5f6", "label": 0}
  ]
}
```

规则与语义：

- **样本**：8～32 个**互异** `order_id`，每个带 0～9 的真值 `label`；
  数量越界/重复 → **400**，标签越界 → **400**，未知单号 → **404**，
  已知但未完成（排队/计算中/取消/过期/失败）→ **409**。
- **拟合**：始终取各单**未校准的原模型分数**（T=1 的 softmax，等价于对原
  logits 做温度缩放），在闭区间 **[0.25, 4.0]** 内最小化平均负对数似然
  （log 空间网格 + 黄金分割精调；NLL 对 1/T 为凸，保证全局最优）。
- **返回**：`current_temperature/current_loss`（现行温度及其在该批样本上的
  NLL）与 `candidate_temperature/candidate_loss`（拟合结果），以及 `applied`
  与生效温度 `temperature`。候选 NLL 未严格更低时 **保留现行值**（`applied=false`）。
- **重叠请求 409**：同一时刻仅允许一个校准请求；拟合在线程池执行，不阻塞
  事件循环，推理与查询照常响应。**拟合失败保持原温度**（返回 500，
  `calibration_in_progress` 自动复位）。
- **生效范围**：校准成功只影响**之后受理**的帧——每个订单在受理时快照当前
  温度（回执含 `temperature` 字段）；排队或计算中的旧单仍用受理时温度，
  已有回执不改写。校准单再次作为样本时仍取未校准分数，可重复提交。

```bash
curl -X POST http://localhost:8000/calibration \
  -H "Content-Type: application/json" \
  -d '{"samples":[{"order_id":"7f6770829f36","label":3}]}'
# {"current_temperature":1.0,"current_loss":2.31,"candidate_temperature":1.84,
#  "candidate_loss":1.97,"applied":true,"temperature":1.84,"sample_count":8}
```

## 规则实现（与需求逐条对应）

- **容量 16**：排队中 + 计算中合计最多 16 张，满额提交返回 429。
- **单批 ≤ 4，等批 ≤ 50ms**：模型空闲时最多等 50ms 凑批，凑满 4 张立即开批。
- **最多一批实际推理**：单个 worker 协程 + 单线程执行器，任何时刻只有一批在做前向计算。
- **排队取消/到期**：从队列移除、释放容量，**不进入模型**（`batch_id` 始终为空）。
- **入批后取消/到期**：立即标记（查询即刻可见）、不供结果；**容量占用到该批计算实际结束**，
  期间不会因此另开一批（worker 严格串行）。
- **批内其他图片正常完成**，分数对应原单；批推理与逐张推理误差 ≤ 1e-5（实测约 7.5e-9）。
- **迟到结果不恢复取消/过期单**：结果回写只针对仍处于 `processing` 且未过期的单。
- **慢推理不挡 API**：阻塞式前向计算放在独立线程，事件循环始终可用。
- **温度校准**：详见上节。拟合在独立校准线程执行，不与推理争用执行器；
  重叠请求 409、拟合失败保持原温度且 API 照常服务；温度按受理时点快照，
  旧单与已有回执不受后续校准影响。

## 配置（环境变量）

| 变量 | 默认 | 说明 |
|---|---|---|
| `MAX_CAPACITY` | 16 | 排队+计算中最大图片数 |
| `BATCH_SIZE` | 4 | 单批最大图片数 |
| `BATCH_WAIT_MS` | 50 | 空闲模型凑批等待上限 |
| `INFERENCE_DELAY_MS` | 0 | **演示用**：在推理线程内注入的人为延迟 |
| `INITIAL_TEMPERATURE` | 1.0 | 首次成功校准前使用的温度（须在 [0.25, 4.0] 内） |
| `MODEL_PATH` | `app/model.pt` | TorchScript 模型路径 |

慢推理演示：`docker compose up` 前把 `docker-compose.yml` 里的 `INFERENCE_DELAY_MS`
改为 `2000`，推理约 2 秒/批，期间提交/查询/取消仍在毫秒级响应（可用 `/stats` 观察）。

## 模型与样图

- `app/model.pt`：固定权重 TorchScript 模型（小型 CNN，10 类，输入 64×64 RGB）。
  由 `scripts/make_model.py` 以固定种子生成，权重 SHA-256 指纹见脚本输出，可复现。
- `samples/`：4 张确定性生成的样图（渐变/圆环/条纹/噪声）。

重新生成：

```bash
python scripts/make_model.py --out app/model.pt
python scripts/make_samples.py --out samples
```

## 测试

```bash
.venv/bin/pytest tests/ -q
```

覆盖：单张完成、批/单张分数一致性（≤1e-5）、凑批窗口、容量 429、排队取消、
入批取消（立即标记+容量仍占用+不另开批）、完成单取消冲突、排队过期、
计算中过期、迟到结果不复活、慢推理下 API 延迟；以及校准 NLL 数学与优化器
（含边界钳制、与暴力扫描一致）、校准生效/不改善保留/恒取未校准分数、
请求校验与状态码（400/404/409）、重叠 409、拟合失败保持原温度、
受理时点温度快照的关键交错、拟合期间 API 不阻塞（`tests/test_calibration.py`）。

## 已知限制

- 订单记录保存在内存中，重启即丢失；未做持久化与历史订单清理（演示规模）。
- 单进程单 worker；多副本扩展需外加负载均衡与共享状态层，不在本交付范围。
