"""
带Streaming但不带速度先验的Baseline配置
用于消融实验的中间baseline

对比实验设计：
1. nusc_baseline_480_60x30_30e.py (无streaming, 无速度) - 原始baseline
2. nusc_baseline_480_60x30_30e_streaming.py (有streaming, 无速度) - 本配置
3. nusc_baseline_480_60x30_30e_velocity.py (有streaming, 有速度) - 完整改进
"""
_base_ = [
    './nusc_baseline_480_60x30_30e.py'
]

# 只启用streaming，不启用速度先验
model = dict(
    head_cfg=dict(
        streaming_cfg=dict(
            streaming=True,           # 启用streaming
            batch_size=1,
            topk=100,                 # baseline用100个query
            trans_loss_weight=5.0,
            use_velocity_prior=False, # 不使用速度先验
        ),
    ),
)


