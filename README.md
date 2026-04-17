# PCA_Quant

`PCA_Quant` 是一个基于任务激活子空间的混合精度量化原型工程。它以 `ASDQ` 的工程组织方式为模板，但将显著性指标替换为 PCA 驱动的子空间对齐信号，用于选择哪些输出通道（权重行）更值得保留高精度。

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
├── main_quant.py                 # 量化入口
├── main_eval.py                  # 评测入口
├── configs/
│   ├── default.yaml
│   └── config.example.yaml
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

### 1. 使用配置文件（推荐）

先在 `configs/default.yaml` 里填好：
- 模型：`model`、`model_args`
- 数据：`data_path`、`image_folder`
- 输出：`scale_path`、`results_path`

然后直接运行：

```bash
python main_quant.py --config configs/default.yaml
```

量化完成后评测：

```bash
python main_eval.py --config configs/default.yaml
```

### 2. 使用命令行直接量化

```bash
python main_quant.py \
  --model llava_onevision \
  --model_args "pretrained=your/model-name" \
  --data_path /path/to/calib.jsonl \
  --image_folder /path/to/images \
  --n_samples 128 \
  --method gate_only \
  --pca_k 32 \
  --w_group 128 \
  --high_precision_ratio 0.1 \
  --high_bit 16 \
  --low_bit 4 \
  --scale_path scale_cache/pca_quant_model.pt \
  --run_process
```

### 3. 评测量化结果（命令行）

```bash
python main_eval.py \
  --config configs/default.yaml \
  --tasks mmmu_val \
  --scale_path scale_cache/pca_quant_model.pt \
  --output_path eval_results
```

## 配置文件

- `configs/default.yaml`
  - 一份可直接修改后运行的默认配置
- `configs/config.example.yaml`
  - 轻量示例配置，可作为复制模板

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
