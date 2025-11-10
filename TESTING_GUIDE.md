# 🧪 StreamMapNet 测试指南

## 📋 目录
- [快速开始](#快速开始)
- [测试命令详解](#测试命令详解)
- [常见场景](#常见场景)
- [结果分析](#结果分析)
- [故障排查](#故障排查)

---

## 🚀 快速开始

### 基本测试命令

```bash
# 多卡测试（推荐）
bash tools/dist_test.sh \
    <CONFIG_FILE> \
    <CHECKPOINT_FILE> \
    <NUM_GPUS> \
    --eval

# 单卡测试
python tools/test.py \
    <CONFIG_FILE> \
    <CHECKPOINT_FILE> \
    --eval
```

---

## 📝 测试命令详解

### 1. **验证集测试（最常用）**

#### 多卡测试（8卡）
```bash
bash tools/dist_test.sh \
    plugin/configs/nusc_newsplit_480_60x30_24e_velocity.py \
    work_dirs/nusc_newsplit_480_60x30_24e_velocity/latest.pth \
    8 \
    --eval
```

#### 单卡测试
```bash
python tools/test.py \
    plugin/configs/nusc_newsplit_480_60x30_24e_velocity.py \
    work_dirs/nusc_newsplit_480_60x30_24e_velocity/latest.pth \
    --eval
```

**说明**：
- `--eval`: 运行评估，计算mAP等指标
- 默认在验证集上测试
- 结果会打印到终端并保存到work_dir

---

### 2. **指定checkpoint测试**

```bash
# 测试特定epoch的checkpoint
bash tools/dist_test.sh \
    plugin/configs/nusc_newsplit_480_60x30_24e_velocity.py \
    work_dirs/nusc_newsplit_480_60x30_24e_velocity/epoch_20.pth \
    8 \
    --eval

# 测试最佳模型
bash tools/dist_test.sh \
    plugin/configs/nusc_newsplit_480_60x30_24e_velocity.py \
    work_dirs/nusc_newsplit_480_60x30_24e_velocity/best_AP_epoch_18.pth \
    8 \
    --eval
```

---

### 3. **保存测试结果**

```bash
# 保存预测结果到pkl文件
bash tools/dist_test.sh \
    plugin/configs/nusc_newsplit_480_60x30_24e_velocity.py \
    work_dirs/nusc_newsplit_480_60x30_24e_velocity/latest.pth \
    8 \
    --eval \
    --work-dir work_dirs/test_results
```

结果会保存在：
- `work_dirs/test_results/results.pkl` - 预测结果
- `work_dirs/test_results/eval_results.txt` - 评估指标

---

### 4. **可视化测试结果**

```bash
# 保存可视化结果
bash tools/dist_test.sh \
    plugin/configs/nusc_newsplit_480_60x30_24e_velocity.py \
    work_dirs/nusc_newsplit_480_60x30_24e_velocity/latest.pth \
    8 \
    --eval \
    --show-dir work_dirs/visualizations
```

可视化图片会保存在 `work_dirs/visualizations/`

---

### 5. **只评估已有结果**

如果已经有预测结果pkl文件，可以直接评估：

```bash
python tools/test.py \
    plugin/configs/nusc_newsplit_480_60x30_24e_velocity.py \
    work_dirs/nusc_newsplit_480_60x30_24e_velocity/latest.pth \
    --result-path work_dirs/test_results/results.pkl
```

---

## 🎯 常见测试场景

### 场景1: 训练完成后立即测试

```bash
# 训练
bash tools/dist_train.sh \
    plugin/configs/nusc_newsplit_480_60x30_24e_velocity.py 8

# 测试（训练完成后）
bash tools/dist_test.sh \
    plugin/configs/nusc_newsplit_480_60x30_24e_velocity.py \
    work_dirs/nusc_newsplit_480_60x30_24e_velocity/latest.pth \
    8 \
    --eval
```

---

### 场景2: 对比多个模型

```bash
# 测试baseline
bash tools/dist_test.sh \
    plugin/configs/nusc_newsplit_480_60x30_24e.py \
    work_dirs/nusc_newsplit_480_60x30_24e/latest.pth \
    8 \
    --eval \
    --work-dir work_dirs/test_baseline

# 测试velocity版本
bash tools/dist_test.sh \
    plugin/configs/nusc_newsplit_480_60x30_24e_velocity.py \
    work_dirs/nusc_newsplit_480_60x30_24e_velocity/latest.pth \
    8 \
    --eval \
    --work-dir work_dirs/test_velocity

# 对比结果
echo "Baseline:"
cat work_dirs/test_baseline/eval_results.txt
echo "\nVelocity:"
cat work_dirs/test_velocity/eval_results.txt
```

---

### 场景3: 测试所有epoch

```bash
# 创建测试脚本
cat > test_all_epochs.sh << 'EOF'
#!/bin/bash

CONFIG="plugin/configs/nusc_newsplit_480_60x30_24e_velocity.py"
WORK_DIR="work_dirs/nusc_newsplit_480_60x30_24e_velocity"

for epoch in {1..24}; do
    echo "Testing epoch $epoch..."
    
    if [ -f "$WORK_DIR/epoch_$epoch.pth" ]; then
        bash tools/dist_test.sh \
            $CONFIG \
            $WORK_DIR/epoch_$epoch.pth \
            8 \
            --eval \
            --work-dir $WORK_DIR/test_epoch_$epoch
        
        echo "Epoch $epoch results:" >> $WORK_DIR/all_epochs_results.txt
        cat $WORK_DIR/test_epoch_$epoch/eval_results.txt >> $WORK_DIR/all_epochs_results.txt
        echo "---" >> $WORK_DIR/all_epochs_results.txt
    fi
done
EOF

chmod +x test_all_epochs.sh
./test_all_epochs.sh
```

---

### 场景4: 测试推理速度

```bash
python tools/benchmark.py \
    plugin/configs/nusc_newsplit_480_60x30_24e_velocity.py \
    work_dirs/nusc_newsplit_480_60x30_24e_velocity/latest.pth
```

**输出示例**：
```
FPS: 12.5
Latency: 80ms
```

---

## 📊 结果分析

### 评估指标说明

测试完成后会输出：

```
+----------------+-------+-------+-------+-------+
|                | AP    | AP_ped| AP_div|AP_bound|
+----------------+-------+-------+-------+-------+
| Results        | 35.3  | 33.2  | 30.5  | 42.0  |
+----------------+-------+-------+-------+-------+
```

**指标含义**：
- **AP (mAP)**: 平均精度，主要指标
- **AP_ped**: 人行横道的AP
- **AP_div**: 车道分隔线的AP
- **AP_bound**: 道路边界的AP

### 详细结果文件

测试后会生成以下文件：

```
work_dirs/nusc_newsplit_480_60x30_24e_velocity/
├── results.pkl              # 预测结果（所有样本）
├── eval_results.txt         # 评估指标（文本格式）
└── test_log.txt            # 测试日志
```

---

## 🔧 高级选项

### 1. 修改测试配置

```bash
# 修改batch size
python tools/test.py \
    plugin/configs/nusc_newsplit_480_60x30_24e_velocity.py \
    work_dirs/nusc_newsplit_480_60x30_24e_velocity/latest.pth \
    --eval \
    --cfg-options data.test.samples_per_gpu=2

# 修改worker数量
python tools/test.py \
    plugin/configs/nusc_newsplit_480_60x30_24e_velocity.py \
    work_dirs/nusc_newsplit_480_60x30_24e_velocity/latest.pth \
    --eval \
    --cfg-options data.workers_per_gpu=4
```

### 2. 使用不同的GPU

```bash
# 指定GPU
CUDA_VISIBLE_DEVICES=0,1,2,3 bash tools/dist_test.sh \
    plugin/configs/nusc_newsplit_480_60x30_24e_velocity.py \
    work_dirs/nusc_newsplit_480_60x30_24e_velocity/latest.pth \
    4 \
    --eval
```

### 3. 调试模式

```bash
# 只测试前10个样本
python tools/test.py \
    plugin/configs/nusc_newsplit_480_60x30_24e_velocity.py \
    work_dirs/nusc_newsplit_480_60x30_24e_velocity/latest.pth \
    --eval \
    --cfg-options data.test.samples_per_gpu=1 \
    | head -100
```

---

## 🐛 故障排查

### 问题1: CUDA out of memory

**错误**：
```
RuntimeError: CUDA out of memory
```

**解决方案**：
```bash
# 方案1: 减少batch size
python tools/test.py \
    ... \
    --cfg-options data.test.samples_per_gpu=1

# 方案2: 使用更少的GPU
bash tools/dist_test.sh ... 4  # 改用4卡

# 方案3: 使用CPU（慢）
CUDA_VISIBLE_DEVICES="" python tools/test.py ...
```

---

### 问题2: checkpoint加载失败

**错误**：
```
FileNotFoundError: checkpoint file not found
```

**解决方案**：
```bash
# 检查checkpoint是否存在
ls -lh work_dirs/nusc_newsplit_480_60x30_24e_velocity/*.pth

# 使用绝对路径
bash tools/dist_test.sh \
    plugin/configs/nusc_newsplit_480_60x30_24e_velocity.py \
    /home/wang/Project/Perception/StreamMapNet/work_dirs/.../latest.pth \
    8 \
    --eval
```

---

### 问题3: 速度信息缺失

**错误**：
```
KeyError: 'velocity'
```

**原因**: 使用了velocity配置但数据集没有速度信息

**解决方案**：
```bash
# 确认使用正确的配置文件
# velocity配置文件会自动计算速度，不需要额外操作

# 如果还是报错，检查数据集代码
python -c "
from mmcv import Config
from mmdet.datasets import build_dataset
cfg = Config.fromfile('plugin/configs/nusc_newsplit_480_60x30_24e_velocity.py')
dataset = build_dataset(cfg.data.test)
data = dataset[0]
print('velocity' in data['img_metas'].data)
"
```

---

### 问题4: 评估结果为0

**可能原因**：
1. 模型未训练好
2. 阈值设置不当
3. 数据集路径错误

**检查方法**：
```bash
# 1. 检查预测结果数量
python -c "
import pickle
results = pickle.load(open('work_dirs/.../results.pkl', 'rb'))
print(f'Total predictions: {len(results)}')
print(f'Sample result: {results[0]}')
"

# 2. 降低阈值重新测试
# 在配置文件中修改 score_thr
```

---

## 📈 性能基准

### 预期测试时间

| 数据集 | 样本数 | 8卡时间 | 单卡时间 |
|--------|--------|---------|----------|
| Newsplit Val | 6019 | ~30分钟 | ~4小时 |
| Baseline Val | 6019 | ~30分钟 | ~4小时 |

### 预期性能

| 模型 | mAP | FPS |
|------|-----|-----|
| Baseline | 63.4 | 12-15 |
| Newsplit | 34.1 | 12-15 |
| Newsplit + Velocity | 35.1-36.1 | 12-15 |

---

## 📝 测试检查清单

在运行测试前，确认：

- [ ] 配置文件路径正确
- [ ] checkpoint文件存在
- [ ] GPU数量正确（1, 2, 4, 8）
- [ ] 数据集路径正确
- [ ] 有足够的磁盘空间（保存结果）
- [ ] 有足够的GPU内存

---

## 🎓 完整测试流程示例

```bash
# 1. 设置变量
CONFIG="plugin/configs/nusc_newsplit_480_60x30_24e_velocity.py"
CHECKPOINT="work_dirs/nusc_newsplit_480_60x30_24e_velocity/latest.pth"
WORK_DIR="work_dirs/test_results"

# 2. 运行测试
echo "开始测试..."
bash tools/dist_test.sh $CONFIG $CHECKPOINT 8 --eval --work-dir $WORK_DIR

# 3. 查看结果
echo "测试完成！结果："
cat $WORK_DIR/eval_results.txt

# 4. 保存结果
cp $WORK_DIR/eval_results.txt results_$(date +%Y%m%d_%H%M%S).txt

echo "结果已保存！"
```

---

## 🔗 相关命令

```bash
# 查看训练日志
tail -f work_dirs/nusc_newsplit_480_60x30_24e_velocity/log.txt

# 查看所有checkpoint
ls -lh work_dirs/nusc_newsplit_480_60x30_24e_velocity/*.pth

# 查看GPU使用情况
watch -n 1 nvidia-smi

# 清理旧的测试结果
rm -rf work_dirs/test_*
```

---

## 💡 最佳实践

1. **总是先测试latest.pth**，确认模型正常
2. **保存测试结果**，方便后续对比
3. **使用多卡测试**，节省时间
4. **定期测试**，监控训练进度
5. **记录所有结果**，方便写论文

---

**祝测试顺利！** 🎉


