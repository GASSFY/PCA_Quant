# SubLink

**SubLink: Flexible Subspace-driven Ultra-low-bit VLM Quantization by Direction-Magnitude Coupling**

This repository implements SubLink, a prototype for mixed-precision VLM quantization driven by activation subspace signals. Quantized weights, evaluation outputs, and logs (e.g. `scale_cache/`, `eval_results/`, `test_result/`, `*.log`) are listed in `.gitignore`; do not commit large artifacts or local secrets.

## Environment

```bash
conda create -n sublink python=3.10 -y
conda activate sublink
cd /path/to/SubLink
pip install -r requirements.txt

# LLaVA-family models (recommended)
git clone https://github.com/LLaVA-VL/LLaVA-NeXT.git
cd LLaVA-NeXT && pip install -e . && cd ..

# lmms-eval (quantization and evaluation)
git clone https://github.com/LSY-noya/lmms-eval.git
cd lmms-eval && pip install -e . && cd ..
```

## Prerequisites (quantization)

```bash
cp configs/default.yaml.example configs/default.yaml
```

Edit `configs/default.yaml` with model paths, calibration data (`data_path`, `image_folder` when using COCO-style calib), and outputs. Set `run_process: true` (or pass `--run_process`) so `main_quant.py` runs calibration and pseudo-quantization; otherwise it only loads an existing `scale_path`.

Method details and scoring definitions are in the paper; this README lists **commands only**.

## Run commands

All commands below assume the **repository root** (`SubLink/`).

### 1) Config-driven quantization (typical)

```bash
python main_quant.py --config configs/default.yaml
```

Evaluation with the same config (override `tasks`, `output_path`, `scale_path` in YAML or on the CLI as needed):

```bash
python main_eval.py --config configs/default.yaml --tasks mmmu_val
```

```bash
python main_eval.py \
  --config configs/default.yaml \
  --tasks mmmu_val \
  --scale_path scale_cache/sublink_model.pt \
  --output_path eval_results
```

### 2) CLI-only quantization

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
  --scale_path scale_cache/sublink_model.pt \
  --run_process
```

If both `--config` and CLI flags are used, YAML keys from `_apply_config` **override** earlier CLI values for the same name—omit a key from YAML if the CLI should win.

### 3) Global hyperparameter search (Optuna: `beta`, `pca_k`)

```bash
pip install -r requirements.txt
```

```bash
python scripts/optuna_tune.py \
  --config configs/default.yaml \
  --tasks mmmu_val \
  --metric-substring mmmu_acc \
  --n-trials 25 \
  --output-dir optuna_study/mmmu_beta \
  --discard-trial-artifacts
```

After search, set `beta` / `pca_k` from `best_params.json` in your YAML and rerun **config-driven quantization** (above) to produce the final checkpoint. Use `--discard-trial-artifacts` to avoid keeping full `.pt` per trial (see script help for behavior).

### 4) Layer-wise search (`beta_l`, `pca_k_l`) and export

```bash
python scripts/optuna_tune_layerwise.py \
  --config configs/default.yaml \
  --n-trials-layer 5 \
  --output-dir optuna_study/layerwise_mse
```

Optional: quantize with a saved layer-wise JSON:

```bash
python main_quant.py \
  --config configs/default.yaml \
  --run_process \
  --layerwise_params optuna_study/layerwise_mse/layerwise_params.json \
  --scale_path scale_cache/layerwise_quant.pt \
  --results_path scale_cache/layerwise_quant.summary.json
```

### Quick reference

| Goal | Command |
|------|---------|
| Quantize with fixed hyperparameters | `python main_quant.py --config configs/default.yaml` |
| Global Optuna search + task eval | `python scripts/optuna_tune.py --config ... --tasks ... --n-trials 25 --output-dir ...` |
| Layer-wise search and export `.pt` | `python scripts/optuna_tune_layerwise.py --config ... --n-trials-layer 5 --output-dir ...` |
| Quantize with existing layer-wise JSON | `python main_quant.py --config ... --run_process --layerwise_params ... --scale_path ...` |

Further tuning hooks are described in [`tune/README.md`](tune/README.md).

## Configuration files

- **`configs/default.yaml.example`** — tracked template; copy to `configs/default.yaml`.
- **`configs/default.yaml`** — local-only (gitignored). Set `model`, `model_args`, calibration paths, and `scale_path` / `results_path` / `output_path` as needed.