# 这是我手动执行的脚本，你能够使用一些不复杂的bash语法，来帮我修改之使其成为一个自动化实验

# 我们需要针对不同的模型，不同的low_bit，不同的量化设置，使用不同的数据集来进行评估

# 模型包括：
MODEL_TYPES=(
  "internvl2"
  "internvl2"
  "llava_onevision"
  "llava_v1.5"
)
MODEL_ARGSS=(
  "pretrained=OpenGVLab/InternVL2-8B"
  "pretrained=OpenGVLab/InternVL2-2B"
  "pretrained=lmms-lab/llava-onevision-qwen2-7b-ov"
  "pretrained=liuhaotian/llava-v1.5-7b"
)


# low_bit的设置包括：2/3/4/6/8
# 量化设置包括：
# high_precision_ratio=0，方法选择gate_only；
# high_precision_ratio=0.01，方法选择gate_only；
# high_precision_ratio=0.01，方法选择abs_only；
# high_precision_ratio=0.01，方法选择proj_log；
# high_precision_ratio=0.1，方法选择gate_only；
# high_precision_ratio=0.1，方法选择abs_only；
# high_precision_ratio=0.1，方法选择proj_log；
# 后面还可能有加
# 数据集包括：mmmu_val、realworldqa、ocrbench、ai2d

# 我希望，统一的量化输出目录为ablation_results，其目录结构为：
# ablation_results/
# ├── model_name/
# │   ├── fp16
# │   ├── low_bit(2/3/4/6/8各一个)/
# │   │   ├── ratio_0/
# │   │   │   ├── gate_only/
# │   │   ├── ratio_0.01/
# │   │   │   ├── gate_only/
# │   │   │   ├── abs_only/
# │   │   │   ├── proj_log/
# │   │   ├── ratio_0.1/
# │   │   │   ├── gate_only/
# │   │   │   ├── abs_only/
# │   │   │   ├── proj_log/

# 到了每个目录下，关于每个数据集的评估结果就不必再分了，使用不同的文件名来区分即可

# 再次强调，不要使用那些复杂的bash语法

# 确定目录排布之后，我要开始告诉你具体应该如何执行了
# 我希望你能够每次更改config，然后执行量化和评估，并保存结果到相应的目录下

# 如何理解呢？我以config来给你解释一下

# ----- 模型 -----
model: llava_onevision
model_args: "pretrained=your/model-name"
batch_size: "1"
device: null

# ----- 校准数据（量化时使用） -----
calib_data: coco
n_samples: 128
data_path: ""
image_folder: ""
interleave_format: false
few_shot_format: false
text_data_path: ""

# ----- PCA mixed precision 方法 -----
# method options: proj_log | proj_norm | gate_only | abs_only
method: gate_only
pca_k: 32
pca_sample_size: 512
beta: 1.0

# ----- 混合精度参数 -----
high_precision_ratio: 0.1
high_bit: 16
low_bit: 4
zero_point: true
# 行内分组大小，当前版本要求对每层 in_features 严格整除
w_group: 128

# ----- 量化输出 -----
run_process: true
pseudo_quant: true
scale_path: "scale_cache/pca_quant_model.pt"
results_path: "scale_cache/pca_quant_model.summary.json"

# ----- 评估参数（main_eval.py 使用） -----
tasks: "mmmu_val"
limit: null
output_path: "eval_results"
seed: "0,1234,1234,1234"
gen_kwargs: ""
log_samples: false
verbosity: "INFO"

# 我们先不说关于Optune的事情，我们先说关于普通量化的事情
# 例如，我们要执行：
# low_bit=2，high_precision_ratio=0.01，方法选择gate_only；

# 先来high_precision_ratio=0，方法选择gate_only。我们先创建对应的目录（以下文字的“对应目录”指的都是此目录），然后，在目录下先复制一份default.yaml
# 你要用python函数去更新我们复制过来的default.yaml，更改model、model_args、method、high_precision_ratio、low_bit、high_precision_ratio
# scale_path设置成对应目录下的scale_cache/pca_quant_model.pt、results_path设置成对应目录下scale_cache/pca_quant_model.summary.json"
# output_path就设置成对应目录下的eval_results
# 运行main_quant.py --config ${对应目录}/default.yaml
# 注意在生成量化checkpoint之后，要执行四个评估任务，这个时候你需要再修改tasks，把四个task都跑完，然后将接入放入到对应目录下的eval_results/results_${task}.json
# 评估main_eval.py --config ${对应目录}/default.yaml
# 然后，在执行量化和评估的时候，需要分别将当前日志打印入对应目录下的logs/quant.log和logs/eval.log
# 然后，执行完四个评估之后，scale_path对应位置的文件即可删除

# 以上是关于gate_only和abs_only的自动化执行过程，我还是需要声明，不要用太复杂的bash语法和python语法

# 然后，我们再来看到proj_log的自动化执行过程，请注意，proj_log是不走main_quant.py的，而是走scripts/optuna_tune_layerwise.py
# 还是先创建对应的目录，然后，在目录下先复制一份default.yaml
# 你要用python函数去更新我们复制过来的default.yaml，更改model、model_args、method、high_precision_ratio、low_bit、high_precision_ratio
# scale_path更改成${对应目录}/layerwise_mse/layerwise_quant.pt，results_path写成""（空字符串）
# output_path还是设置成对应目录下的eval_results
# 接下来，你需要运行
python scripts/optuna_tune_layerwise.py \
  --config ${对应目录}/default.yaml \
  --n-trials-layer 5 \
  --output-dir ${对应目录}/layerwise_mse
# 上面的脚本执行完毕之后，要执行四个评估任务，这个时候你需要再修改tasks，把四个task都跑完，然后将接入放入到对应目录下的eval_results/results_${task}.json
# 评估main_eval.py --config ${对应目录}/default.yaml
# 运行完四个评估之后，只需删除${对应目录}/layerwise_mse/layerwise_quant.pt即可，其他的不用动

# 以上是关于proj_log的自动化执行过程，我还是需要声明，不要用太复杂的bash语法和python语法

