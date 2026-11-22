# 图纸智析 · 初赛提交物

- 队伍：vitality
- 赛题：第二届北京市「人工智能+」创新大赛 · 赛题 4「图纸智析」

## 1 模型说明

离线三段式流水线，默认 `--mode blend`，随包三个权重文件：

1. **检测与分割基座**：Mask R-CNN R50-FPN v2（COCO 预训练初始化，36 类输出），在 `train_final` 的部分记录上微调；ROI 掩码头由 28×28 换成 112×112 并继续微调 550 步。权重 `weights/fine_mask112_u550.pt`（193,126,801 B，sha256 `cf6c1fe68e7f0c626428f5ae6975a0c0abd4acd8462f3daa19fb23d333e458ba`）。
2. **候选池与墨迹吸附**：掩码阈值 0.40、检测分数 ≥ 0.50、每图最多 300 个候选；再以 PNG 自身的笔画连通域为约束把预测掩码吸附到墨迹上（合成分辨率 128），把亚像素偏移的掩码拉回 IoU > 0.5 区间。
3. **筛选器**：每个候选抽取 26 维特征，送入 `weights/blend_head_20261120.pkl`（3 个 HistGradientBoosting 与 3 个 ExtraTrees 各占 1/6 概率权重；在 256² 笔画头掩码上重标标签后重训）。概率 ≥ 0.70 保留；同类框 IoU ≥ 0.10 的候选按概率降序贪心抑制；每图最多 60 个实例；Stuff 类（≥ 30）不发射。
4. **笔画头**：每个被选中的实例按 `max(边长, 64) + 8` px 的窗口裁图、重采样到 256²，由 256² U-Net（width 32，4 输入通道：墨迹灰度 / 框掩码 / u / v，1,948,577 参数）输出中心线概率，取「概率 ≥ 0.70 且墨迹灰度 ≥ 96」得 1 px 中心线掩码，替换该实例的 112² 掩码。权重 `weights/mask_head256_u1.pt`（7,809,695 B，sha256 `9b03eee11953f42d3b1162974211ac6925c1db6fa806627f734a8f57106e9883`）。默认 `--mask-source head`；`--mask-source mask` 回到 112² 掩码行为。

推理实际加载参数 54,591,637（0.055B，占 5B 上限的 1.09%），详见同目录 `参数统计.md`。

## 2 运行环境

- Python 3.12；PyTorch 2.14.1+cu130；torchvision 0.29.1+cu130；其余依赖见 `requirements.txt`。
- 有 GPU 时用 `--device cuda:0`，纯 CPU 亦可运行。
- 离线运行，不联网，不调用任何外部模型 API；容器见 `Dockerfile`。
- GPU 运行需 CUDA 版 torchvision：CPU wheel 会在 `torchvision::nms` 报无 CUDA 后端。可用 `--tv-overlay` 或环境变量 `CAD_TV_OVERLAY_DIR` 指向同版本 cu130 wheel 的 `pip install --target` 目录。

## 3 完整推理命令

在包根目录执行（`infer.py`、`delivery/`、`weights/` 均在当前目录）：

```bash
python infer.py \
  --inputs <PNG 目录> \
  --out <新的空目录> \
  --device cpu \
  --mode blend \
  --verify
```

`--checkpoint`、`--blend-model` 与 `--mask-head-ckpt` 的默认值分别是 `weights/fine_mask112_u550.pt`、`weights/blend_head_20261120.pkl` 与 `weights/mask_head256_u1.pt`（相对包根解析），可省略。`blend` 默认带 `--mask-source head --mask-head-min-side 0 --mask-head-tau 0.70`，即每个被选中实例都由 256² 笔画头重画掩码；加 `--mask-source mask` 回到 112² 掩码。

- `--inputs` 可以是 PNG 目录、`inputs.json` 清单或单张 PNG；`--out` 必须是空目录或不存在。
- 产物：`<out>/results/<stem>.json`（交付文档，文件名与输入 PNG 的 stem 一致）、`<out>/<stem>.json`（逐份平铺副本，`--no-flat` 可关）、`<out>/infer_report.json` 等运行记录；`--verify` 追加跑 `delivery/verify_delivery.py` 并写 `run_receipt.json`。
- 长跑中断后，原命令加 `--resume`、沿用同一 `--out` 续跑：已写好的文档跳过，只补缺。

## 4 随机种子与可复现性

- 推理全程无随机采样：权重文件固定（sha256 `cf6c1fe6…`、`5ab1ad03…`、`9b03eee1…`），筛选器为确定性树遍历，同类框抑制按概率降序、平局保持候选顺序。
- 输出文档不含时间戳等易变字段。
- 不同 torch / CUDA 版本之间，个别处于掩码阈值或分组边界的候选可能翻转，故跨环境复现以「12 字段契约合规、指标同级」为准，不承诺图元级逐位一致。
- 训练侧种子：掩码头微调 `seed 20261108`；筛选器 3 个随机种子 × 2 类学习器。

## 5 交付内容

| 项 | 位置（包根） | 说明 |
| --- | --- | --- |
| 预测 JSON | `results/` | 1000 份，与 `dev_final` 的 1000 张 PNG 五位 stem 一一对应；737,777 图元 / 4,118 实例 |
| 统一入口 | `infer.py` | PNG → 12 字段 JSON，自带 `--verify` |
| 依赖 / 容器 | `requirements.txt`、`Dockerfile` | 离线，无外网 API |
| 权重 | `weights/fine_mask112_u550.pt`、`weights/blend_head_20261120.pkl`、`weights/mask_head256_u1.pt` | sha256 见 `MANIFEST.sha256` |
| 方法说明 | `方法说明.md` | 3 页内 |
| 参数统计 | `参数统计.md` | 54,591,637（5B 上限的 1.09%） |
