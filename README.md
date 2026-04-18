# PCA_Quant

`PCA_Quant` 是一个基于任务激活子空间的混合精度量化原型工程。它以 `ASDQ` 的工程组织方式为模板，但将显著性指标替换为 PCA 驱动的子空间对齐信号，用于选择哪些输出通道（权重行）更值得保留高精度。

运行产生的量化权重、评测结果与日志（如 `scale_cache/`、`eval_results/`、`test_result/`、`*.log`）已由 `.gitignore` 排除，请勿将大文件或本地配置提交进仓库。

## 环境构建

```bash
# 1) 创建并激活环境
conda create -n pca_quant python=3.10 -y
conda activate pca_quant

# 2) 进入项目目录
cd E:\LLM-learning\PCA_Quant

# 3) 安装项目依赖
pip install -r requirements.txt

# 4) 安装 LLAVA-NeXT 依赖（LLaVA 系模型推荐）
cd /root/autodl-tmp
git clone https://github.com/LLaVA-VL/LLaVA-NeXT.git
cd LLaVA-NeXT
pip install -e .

# 5) 安装 lmms-eval（量化/评测入口都依赖）
git clone https://github.com/LSY-noya/lmms-eval.git
cd lmms-eval
pip install -e .
```

## 方法概览

当前版本支持四种输出通道打分方法，通过 `--method` 切换：

- `proj_log`：`Gate + beta * log(Abs + eps)`
- `proj_norm`：`Gate + beta * Z(Abs)`（`Z` 为全 Transformer 全局 z-score）
- `gate_only`：仅使用 Gate
- `abs_only`：仅使用 Abs

其中：

- `Gate_i = ||P_k w_i||_2 / (||w_i||_2 + eps)`
- `Abs_i = ||P_k w_i||_2`

这里 `w_i` 是第 `i` 个输出通道对应的权重行向量，`P_k = U_k U_k^T`，`U_k` 是该层输入激活 PCA 的前 `k` 个主成分基底。`epsilon` 只作为函数内部数值稳定项，不作为命令行参数暴露。

## 架构

```text
PCA_Quant/
├── main_quant.py                 # 量化入口（校准 + PCA + 选通道 + 伪量化 + 保存权重）
├── main_eval.py                  # 评测入口（lmms-eval）
├── scripts/
│   └── optuna_tune.py            # 可选：用 Optuna（TPE）搜索 beta，复用一次 PCA 统计
├── tune/                         # 搜索空间与指标解析（见 tune/README.md）
├── configs/
│   ├── default.yaml.example   # 仓库内模板（会提交）
│   └── default.yaml           # 本地配置（不提交，从模板复制）
└── PCA/
    ├── calibration/
    │   └── coco.py              # 校准数据读取
    ├── models/                  # 各模型包装器
    ├── quantization/
    │   ├── activation_collector.py
    │   ├── mixed_precision.py
    │   ├── pca_importance.py
    │   ├── quant_funcs.py
    │   └── quantize.py
    └── utils/
```

量化主流程：

1. 通过模型 wrapper 和校准数据构建前向 mini-batches。
2. 收集每层线性层输入激活，估计 `E[x^2]`，并对采样激活做 PCA。
3. 根据 `--method` 计算输出通道重要性分数。
4. 按 `--high-precision-ratio` 在每层独立选取 top-k 高精度通道（各层单独排序，不做跨层竞争）。
5. 对权重执行“行内分组量化 + 保留输出行高 bit 量化”的混合精度伪量化，并保存量化权重。

## 核心参数

### 量化相关

- `--method`
  - 可选：`proj_log`、`proj_norm`、`gate_only`、`abs_only`
- `--pca_k`
  - PCA 主成分个数
- `--pca_sample_size`
  - 每个线性层最多保留多少 token 参与 PCA 估计
- `--high_precision_ratio`
  - 每层独立保留为高精度的输出通道（权重行）比例
- `--high_bit`
  - 高精度通道的量化 bit，默认 `16`
- `--low_bit`
  - 低精度通道的量化 bit，默认 `4`
- `--beta`
  - 用于 `proj_log` 和 `proj_norm`
- `--w_group`
  - 行内分组大小（group size），当前版本采用严格整除策略
- `--scale_path`
  - 量化权重保存路径
- `--results_path`
  - 量化摘要 json 保存路径

### 数据与模型

- `--model`
  - 当前 wrapper 注册名，如 `llava_onevision`、`llava`、`internvl2`
- `--model_args`
  - 透传给 `lmms-eval` 模型构造器的参数字符串
- `--data_path`
  - 校准数据 json/jsonl
- `--image_folder`
  - 图像根目录
- `--n_samples`
  - 校准样本数

## 运行命令

以下命令均在**项目根目录**（`PCA_Quant/`）下执行。量化入口始终是 **`main_quant.py`**；更新仓库后，若你仍用 `python main_quant.py --config ...` 与原先一致，**行为不变**（`--beta` 仍从配置文件或命令行传入）。

### 执行量化前需要满足什么

1. **本地配置**：若还没有 `configs/default.yaml`，先从模板复制并填写路径与模型参数（该文件默认被 Git 忽略）：

   ```bash
   cp configs/default.yaml.example configs/default.yaml
   ```

2. **量化开关**：在 YAML 中设置 `run_process: true`（或命令行加 `--run_process`），否则 `main_quant.py` 只做「加载已有 `scale_path`」分支，不会执行校准与伪量化。

3. **校准数据**：`calib_data: coco` 时需填写 `data_path` 与 `image_folder`（与 `main_quant` 要求一致）。

4. **`beta` 何时生效**：仅当 `method` 为 `proj_log` 或 `proj_norm` 时，`beta` 会影响通道重要性；`gate_only` / `abs_only` 下可忽略 `beta`。

---

### 方式一：仅用配置文件完成量化（最常用）

在 `configs/default.yaml` 中填好模型、校准数据、`scale_path`、`results_path`，以及方法、`beta`（若使用 `proj_log` / `proj_norm` ）等字段后执行：

```bash
python main_quant.py --config configs/default.yaml
```

量化完成后，用同一份配置做评测（请保证 YAML 里 `tasks`、`output_path`、`scale_path` 等与评测一致，或命令行显式覆盖）：

```bash
python main_eval.py --config configs/default.yaml --tasks mmmu_val
```

若评测相关项未写在 YAML 里，可在命令行补全，例如：

```bash
python main_eval.py \
  --config configs/default.yaml \
  --tasks mmmu_val \
  --scale_path scale_cache/pca_quant_model.pt \
  --output_path eval_results
```

---

### 方式二：命令行直接量化（不依赖 YAML 中的部分字段）

与过去相同，可显式传入 `--model`、`--data_path`、`--beta`、`--run_process` 等，例如：

```bash
python main_quant.py \
  --model llava_onevision \
  --model_args "pretrained=your/model-name" \
  --data_path /path/to/calib.jsonl \
  --image_folder /path/to/images \
  --n_samples 128 \
  --method proj_log \
  --beta 1.0 \
  --pca_k 32 \
  --w_group 128 \
  --high_precision_ratio 0.1 \
  --high_bit 16 \
  --low_bit 4 \
  --scale_path scale_cache/pca_quant_model.pt \
  --run_process
```

若同时使用 `--config` 与上述参数，合并顺序为：先解析命令行 `parse_args()`，再对 YAML 中的每一项执行 `_apply_config`。**因此 YAML 里出现的键会覆盖命令行里已传入的同名参数**（若希望以命令行为准，请勿在 YAML 中写该键）。

---

### 方式三：用 Optuna 搜索 `beta` 后再做「最终」量化（可选）

适用场景：希望在固定任务指标（如 `mmmu_val` 的 `mmmu_acc`）下自动搜索 `beta`，再把最优值用于一次正式量化。

**步骤 1 — 安装依赖**（`requirements.txt` 已包含 `optuna`）：

```bash
pip install -r requirements.txt
```

**步骤 2 — 配置**：在 YAML 中设置 `method: proj_log` 或 `proj_norm`，并 `run_process: true`；校准与模型路径需正确。

**步骤 3 — 运行搜索**（在仓库根目录）：

```bash
python scripts/optuna_tune.py \
  --config configs/default.yaml \
  --tasks mmmu_val \
  --metric-substring mmmu_acc \
  --n-trials 20 \
  --output-dir optuna_study/mmmu_beta
```

说明：

- 脚本会**只做一次**校准与 PCA 统计，每个 trial 仅更换 `beta` 并重跑「重要性 → 选通道 → 伪量化 → 评测」，比重复跑完整 `main_quant` 全流程更省时间。
- 结果目录下会生成 **`best_params.json`**（含最优 `beta`）以及各 trial 的 checkpoint；`optuna_study/` 已在 `.gitignore` 中，避免误提交。

**步骤 4 — 将最优 `beta` 写回配置并执行最终量化**：

1. 打开 `optuna_study/mmmu_beta/best_params.json`，记下 `best_params.beta`（或 `best_tune_hyperparams` 中的值）。
2. 编辑 `configs/default.yaml`，将 `beta` 设为该值（并保持 `method` 仍为 `proj_log` 或 `proj_norm`）。
3. 设定你希望的最终权重路径，例如 `scale_path: "scale_cache/pca_quant_model.pt"`，再运行：

   ```bash
   python main_quant.py --config configs/default.yaml
   ```

4. 如需全量评测，再运行 `main_eval.py`（同上）。

可选：搜索阶段为加快评测可在 YAML 中设较小的 `limit`；最终量化与评测可改回全量 `limit` 或去掉限制。

---

### 小结：我该用哪条命令「执行量化」？

| 目的 | 命令 |
|------|------|
| 日常量化（固定 `beta`） | `python main_quant.py --config configs/default.yaml` |
| 调参搜索 `beta` | `python scripts/optuna_tune.py --config ... --tasks ... --output-dir ...` |
| 搜索完成后，用最优 `beta` 再训一次权重 | 把 `beta` 写入 YAML 后，再次 `python main_quant.py --config configs/default.yaml` |

扩展搜索维度（除 `beta` 外更多超参）时，见 [`tune/README.md`](tune/README.md)。

## 配置文件

- `configs/default.yaml.example`
  - 仓库内提交的模板；更新依赖或加新字段时以它为准
- `configs/default.yaml`
  - 本地运行用配置，从模板复制后自行修改，**不会被 Git 跟踪**（与 `ASDQ` 的 `configs/default.yaml` + `default.yaml.example` 约定一致）

推荐先只改三类字段：

- 模型：`model`、`model_args`
- 数据：`data_path`、`image_folder`
- 输出：`scale_path`、`results_path`、`output_path`

## 当前版本说明

当前版本重点是先跑通：

- PCA 统计收集
- 四种方法的统一接口
- 混合精度量化入口
- 基础评测入口
- 配置和文档说明

分组策略说明（当前固定）：

- 混合精度量化路径采用行内分组（按输入维度分段）。
- 保留对象是输出通道（权重行），选中行使用高 bit 量化，其他行使用低 bit 量化。
- 仅支持严格整除：每个线性层都需满足 `in_features % w_group == 0`。
- 如果不满足，会在量化前报错并给出该层可选的整除因子建议。

后续可以继续扩展：

- 更稳定的 PCA 采样/增量估计
- 更丰富的结果保存
- 参数搜索
- 更系统的对比实验脚本
