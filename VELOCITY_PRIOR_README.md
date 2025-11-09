# 速度先验改进方案

## 📝 概述

本改进方案为StreamMapNet添加了**速度先验**功能，通过融合自车运动信息来提升地图元素的时序预测能力。

### 核心思想
- 从GPS/IMU位姿变化计算精确速度
- 将速度信息作为额外先验注入到query更新模块
- 帮助模型理解物体的运动趋势，减少学习难度

### 预期收益
- **整体AP提升**: +1.0-2.0
- **时序一致性**: 显著改善
- **高速场景**: 鲁棒性提升

---

## 🏗️ 实现架构

### 1. 数据层 (`plugin/datasets/nusc_dataset.py`)
- 新增 `_compute_velocity()` 方法
- 从相邻帧位姿计算速度
- 在ego坐标系下表示

### 2. 模型层 (`plugin/models/utils/velocity_motion_mlp.py`)
- `VelocityMotionMLP`: 基础版本，拼接位姿和速度编码
- `AdaptiveVelocityMotionMLP`: 高级版本，自适应融合权重

### 3. Head层 (`plugin/models/heads/MapDetectorHead.py`)
- 集成速度编码到query propagation
- 通过配置开关控制是否启用

---

## 🚀 使用方法

### 方法1: 使用预配置文件

```bash
# 训练
bash tools/dist_train.sh plugin/configs/nusc_newsplit_480_60x30_24e_velocity.py 8

# 测试
bash tools/dist_test.sh plugin/configs/nusc_newsplit_480_60x30_24e_velocity.py work_dirs/nusc_newsplit_480_60x30_24e_velocity/latest.pth 8 --eval
```

### 方法2: 修改现有配置

在任意配置文件中添加：

```python
model = dict(
    pts_bbox_head=dict(
        streaming_cfg=dict(
            streaming=True,
            batch_size=1,
            topk=300,
            trans_loss_weight=5.0,
            use_velocity_prior=True,  # 启用速度先验
        ),
    ),
)
```

### 方法3: 对比实验（消融研究）

```bash
# 基线（无速度先验）
bash tools/dist_train.sh plugin/configs/nusc_newsplit_480_60x30_24e.py 8

# 带速度先验
bash tools/dist_train.sh plugin/configs/nusc_newsplit_480_60x30_24e_velocity.py 8

# 对比结果
python tools/analysis/compare_results.py \
    --baseline work_dirs/nusc_newsplit_480_60x30_24e/results.pkl \
    --velocity work_dirs/nusc_newsplit_480_60x30_24e_velocity/results.pkl
```

---

## 📊 速度信息说明

### 数据格式

每个样本的 `img_metas` 中包含：

```python
{
    'velocity': [vx, vy, vz],  # ego坐标系下的速度 (m/s)
    'velocity_magnitude': float,  # 速度大小 (m/s)
    'timestamp': int,  # 时间戳 (微秒)
}
```

### 速度编码

传入模型的速度编码为4维向量：

```python
velocity_encoding = [vx, vy, |v|, dt]
```

其中：
- `vx, vy`: ego坐标系下的x, y方向速度
- `|v|`: 速度大小 (xy平面)
- `dt`: 与上一帧的时间间隔

---

## 🔬 技术细节

### 速度计算方法

```python
# 1. 全局坐标系下的位移
global_displacement = pos_next - pos_curr

# 2. 全局速度
global_velocity = global_displacement / dt

# 3. 转换到ego坐标系
ego_velocity = R_ego2global.T @ global_velocity
```

### 坐标系说明

```
Global (世界坐标系)
  ↓ ego2global
Ego (自车坐标系) ← 速度在这里
  ↓ sensor2ego
Sensor (传感器坐标系)
```

### Query更新流程

```
上一帧query → [位姿编码(12维) + 速度编码(4维)] → MLP → 更新后query
```

---

## ⚙️ 配置参数

### streaming_cfg 参数

| 参数 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `use_velocity_prior` | bool | False | 是否启用速度先验 |
| `streaming` | bool | True | 是否启用streaming |
| `batch_size` | int | 1 | batch大小 |
| `topk` | int | 300 | 保留的top-k queries |
| `trans_loss_weight` | float | 5.0 | 传播loss权重 |

---

## 📈 实验建议

### 阶段1: 基础验证 (1-2周)
1. 使用默认配置训练
2. 验证是否有正向收益
3. 分析不同场景的表现

**预期结果**: +0.5-1.0 AP

### 阶段2: 参数调优 (1-2周)
1. 调整学习率
2. 调整trans_loss_weight
3. 尝试不同的速度编码方式

**预期结果**: +1.0-1.5 AP

### 阶段3: 高级改进 (2-3周)
1. 使用 `AdaptiveVelocityMotionMLP`
2. 加入角速度信息
3. 实现注意力调制

**预期结果**: +1.5-2.5 AP

---

## 🐛 故障排查

### 问题1: 训练不收敛

**可能原因**: 速度编码的数值范围与位姿编码不匹配

**解决方案**:
```python
# 在velocity_motion_mlp.py中添加归一化
velocity_encoding = velocity_encoding / velocity_encoding.abs().max()
```

### 问题2: 静止场景性能下降

**可能原因**: 静止时速度为0，提供的信息有限

**解决方案**:
```python
# 在MapDetectorHead.py中添加判断
if velocity_magnitude < 0.5:  # 静止
    use_velocity_prior = False
```

### 问题3: 速度信息为None

**可能原因**: 数据加载时未正确计算速度

**检查方法**:
```python
# 在训练脚本中添加
print("Sample keys:", data['img_metas'][0].keys())
print("Velocity:", data['img_metas'][0].get('velocity'))
```

---

## 📚 参考文献

### 相关工作
1. **BEVFormer**: 使用can_bus信息增强BEV特征
2. **FIERY**: 使用速度预测未来轨迹
3. **MotionNet**: 运动信息用于3D检测

### 理论基础
- 刚体运动学
- 相对运动原理
- 卡尔曼滤波

---

## 🤝 贡献

如果你有改进建议或发现问题，欢迎：
1. 提交Issue
2. 创建Pull Request
3. 分享实验结果

---

## 📄 许可证

与StreamMapNet主项目保持一致

---

## 📧 联系方式

如有问题，请通过以下方式联系：
- GitHub Issues
- Email: [your-email]

---

## 🎉 致谢

感谢StreamMapNet原作者提供的优秀基础框架！

---

**最后更新**: 2025-11-09
**版本**: v1.0
**状态**: ✅ 已实现，待测试

