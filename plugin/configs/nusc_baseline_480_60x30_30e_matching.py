"""
Baseline + 点匹配先验配置
基于 nusc_baseline_480_60x30_30e.py

注意：同时启用streaming和点匹配先验
"""
_base_ = [
    './nusc_baseline_480_60x30_30e.py'
]

# 启用streaming和点匹配先验
model = dict(
    head_cfg=dict(
        streaming_cfg=dict(
            streaming=True,
            batch_size=1,
            topk=100,  # baseline用100个query
            trans_loss_weight=5.0,
            use_velocity_prior=False,
            use_point_matching_prior=True,     # 🆕 启用点匹配先验
            matching_loss_weight=0.5,          # 🆕 匹配loss权重
        ),
    ),
)

