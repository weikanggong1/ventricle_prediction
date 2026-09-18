# Ventricle prediction — 训练、测试与预训练模型

将 `train_transformer_t1_orig_masked_20260206.ipynb` 和 `train_transformer_t1_orig_Width_20260206.ipynb` 整合为一套可直接运行的训练/测试代码。

- **masked 模型**：预测 EI、z-EI、BVR_AC、BVR_PC，使用 `all_ven.nii.gz`。
- **width 模型**：预测 Width，使用 `inf_ven.nii.gz`。
- 两个模型分别随机初始化、训练和选模，最终汇总五项指标。
- 支持从零训练，也提供 UKB 与 ADNI 的小模型 pretrained checkpoints。仓库不含原始受试者影像、标注或个体预测。

## Pretrained checkpoints

提供四份从随机初始化、在 CPU 上重新训练得到的权重：

| 数据集 | 四指标模型 | Width 模型 |
|---|---|---|
| UKB | [masked.pt](pretrained/ukb/masked.pt) | [width.pt](pretrained/ukb/width.pt) |
| ADNI | [masked.pt](pretrained/adni/masked.pt) | [width.pt](pretrained/adni/width.pt) |

独立测试集表现、均值预测基线、模型参数量和训练细节见 [PERFORMANCE.md](PERFORMANCE.md)。

```bash
python pipeline.py predict --manifest subjects.csv --checkpoints pretrained/ukb --output ukb_predictions.csv --device cpu
python pipeline.py predict --manifest subjects.csv --checkpoints pretrained/adni --output adni_predictions.csv --device cpu
```

公开权重包含模型、掩膜、体素排序和标准化参数，不含受试者 ID 或影像路径。公开权重用于 `predict`；`evaluate` 的重叠检查需要自行训练产生的私有权重。仓库报告中的测试是在私有权重上完成，并已核对公开权重重载预测与之完全一致。

## 1. 安装

```bash
pip install -r requirements.txt
```

已在 Python 3.11、PyTorch 2.5.1 环境验证。默认 CPU；本次公开权重均在 CPU 上训练。

## 2. 准备数据清单

假设影像和标注已经存在，准备一个 CSV：

```csv
eid,image,EI,z-EI,BVR_AC,BVR_PC,Width
subject_001,/data/subject_001/T1_brain_linear_1mm.nii.gz,0.30,0.25,1.10,1.20,8.0
subject_002,/data/subject_002/T1_brain_linear_1mm.nii.gz,0.32,0.27,1.15,1.25,8.5
```

以上数值仅演示格式。`eid` 作为字符串读取；`image` 可为绝对路径，或相对于 CSV 的路径。每位受试者一行，ID 和影像路径不能重复。

支持原 notebook 的六位标注者格式，例如 `EI_reader1` 至 `EI_reader6`、`Width_reader1` 至 `Width_reader6`：程序按每项非缺失标注的均值作为标签。如直接标签列和 reader 列同时存在，优先使用直接标签。训练及评估所需目标不能全部缺失。

影像应已完成与原 notebook 一致的线性配准及预处理；本程序不负责从原生空间影像生成配准结果。随包掩膜网格为 `(182, 218, 182)`、1 mm，affine 为：

```text
-1  0  0   90
 0  1  0 -126
 0  0  1  -72
 0  0  0    1
```

加载时检查图像和掩膜的 shape/affine。配准质量仍需在数据准备阶段确认。

## 3. 一条命令完成训练和测试

```bash
python pipeline.py run \
  --manifest /path/to/subjects.csv \
  --output runs/experiment_01 \
  --device cuda:0
```

这条命令依次完成：

1. 划分 train/val/test，保存 `split_manifest.csv`。
2. 只在训练集计算体素排序和标签均值/标准差。
3. 从随机初始化训练两个模型，以验证集平均 Pearson r 选择最佳模型。
4. 重新加载两份最佳模型，在独立测试集评估并汇总五指标。

未提供 `split` 列时，默认 seed=42，约 70% 训练、15% 验证、其余测试，各集合至少 2 人。也可指定数量，例如 250 例分为 200/25/25：

```bash
python pipeline.py run --manifest subjects.csv --train-count 200 --val-count 25 \
  --output runs/experiment_02 --device cuda:0
```

如已有分组，在 CSV 加 `split` 列，取值 `train`、`val`、`test`；程序直接采用既有划分。重复扫描和有关联的样本应在生成清单时按受试者/家系等分组，不能跨集合。若 CSV 已含 split，命令行数量参数不改变它。

默认训练 50 epochs，5 epochs warmup，batch size=4，AdamW lr=1e-4，input dropout=0.25。`--model-size small` 使用 patch=512、depth=4（5 blocks）、embed_dim=64、heads=4；默认 `original` 保留原结构。可指定 `--epochs`、`--warmup-epochs`、`--batch-size`、`--workers`、`--lr`、`--seed`。模型保留原 notebook 的 patch=256、depth=18、embed_dim=128、heads=8；depth=18 是原 UViT 构造参数，对应 9+1+9 个 block。

`run` 不加载已有权重；输出目录若已有模型则要求改用新目录。

### 输出

```text
runs/experiment_01/
├── split_manifest.csv
├── checkpoints/
│   ├── masked.pt
│   ├── width.pt
│   ├── masked_history.json
│   └── width_history.json
├── test_predictions.csv
├── test_predictions.metrics.json
└── test_predictions.provenance.json
```

- `test_predictions.csv`：eid，以及每项预测和对应的 `_true` 标签列。
- `metrics.json`：每项的 n、MAE、RMSE、Pearson r、R²，按原始标签尺度计算；相关或 R² 无定义时记录为 `null`。
- 权重文件保存模型参数、掩膜/affine、体素顺序、标签标准化参数、目标顺序、最佳 epoch 以及训练/验证 ID 和影像路径。独立测试会检查与这些 ID/路径是否重叠。

### 复现本次 CPU 小模型训练

```bash
python prepare_cohort.py --cohort ukb --labels /data/result_round123_250samples.csv --image-root /data/ukb --output ukb_manifest.csv
python prepare_cohort.py --cohort adni --labels /data/ADNI_R3_200samples.txt --image-root /data/adni --output adni_manifest.csv

OMP_NUM_THREADS=8 MKL_NUM_THREADS=8 OPENBLAS_NUM_THREADS=8 CUDA_VISIBLE_DEVICES= python pipeline.py run --manifest ukb_manifest.csv --output runs/ukb_small --model-size small --device cpu --batch-size 8 --epochs 50
OMP_NUM_THREADS=8 MKL_NUM_THREADS=8 OPENBLAS_NUM_THREADS=8 CUDA_VISIBLE_DEVICES= python pipeline.py run --manifest adni_manifest.csv --output runs/adni_small --model-size small --device cpu --batch-size 8 --epochs 50
```

UKB 图像路径为 `<image-root>/<eid>/T1_brain_linear_1mm.nii.gz`；ADNI 为 `<image-root>/<PTID>/T1_MNI.nii.gz`。UKB 取六位标注者的非缺失均值，ADNI 取 R3 表中两位标注者的均值。脚本保留原始表行顺序，以 seed=42 划分，并记录缺失影像/标签的数量。训练时将掩膜内向量载入内存一次，避免每个 epoch 重复解压 NIfTI。

## 4. 分开运行训练、测试和预测

训练时清单需要已有 train/val/test 三组；`run` 会自动生成这种清单。

```bash
# 从零训练；只读取 train/val 的影像和标签用于拟合与选模
python pipeline.py train --manifest split_manifest.csv --output runs/manual/checkpoints --device cuda:0

# 独立测试；含 split 时仅使用 test 行，也可传入另一份纯测试集清单
python pipeline.py evaluate --manifest split_manifest.csv \
  --checkpoints runs/manual/checkpoints --output runs/manual/test.csv --device cuda:0

# 无标签新影像的预测；只需 eid,image，不做评估
python pipeline.py predict --manifest new_subjects.csv \
  --checkpoints runs/manual/checkpoints --output runs/manual/predictions.csv --device cuda:0
```

默认两个任务均运行。`--task masked` 或 `--task width` 可只训练/评估其中一项。新影像预测清单无需与训练清单采用同一目录布局。

## 5. 小型示例：从零跑通

`example/` 提供 20 个完全模拟的小型 NIfTI 和标签，无已有权重。它们是随机数组流程示例，不是解剖学模拟数据，不能用于衡量实际预测性能。

```bash
python pipeline.py run --manifest example/subjects.csv --masks example/masks \
  --train-count 12 --val-count 4 --epochs 2 --warmup-epochs 1 \
  --output runs/example
```

也可以重新生成示例（seed=20260918）：

```bash
python make_example.py --output my_example
```

## 6. 验证

```bash
python -m unittest -v test_pipeline.py
python smoke_test.py --output smoke_run --model-size small
```

已经在 gpucw1 的 CPU 上通过：

- 7 项测试：图像标准化/网格、异常输入、标签与重复 ID、模型重载、数据划分/泄漏检查、评估指标，以及去除私有元数据后的权重导出一致性。
- 两个完整结构模型从随机初始化各训练 2 epochs；模拟样本分为 12 训练、4 验证、4 测试。
- 最佳模型在新进程重载后，独立测试输出完全一致。
- 训练数据被用作独立测试时会拒绝，并且不写出测试结果。

见 `validation/smoke_report.json`（原结构）、`validation/small_smoke_report.json`（小模型）和 `validation/unit_tests.txt`。模拟验证针对执行链；真实 UKB/ADNI 重新训练与测试结果见 [PERFORMANCE.md](PERFORMANCE.md)。

## 与原 notebook 的对应关系

保留原 UViT 前向和多目标加权回归损失；两个掩膜独立使用；每影像在掩膜内用总体标准差标准化；体素按训练集“各目标绝对相关均值”排序。标签参与相关计算前的 float32 转换也保留。目标 CA 未在这两份 notebook 中训练，因此本代码不输出 CA。

原实现中的坐标编码没有参与实际前向，本版保持原计算方式；移除了模型模块导入时自动执行的演示。验证改为每个 epoch 结束执行，增加了独立测试集；测试数据不参与标准化统计、体素排序或最佳模型选择。
