"""
带速度先验的Baseline配置
基于 nusc_baseline_480_60x30_30e.py

注意：Baseline版本默认没有启用streaming，这个配置会同时启用streaming和速度先验
如果只想测试速度先验的效果，建议使用已经有streaming的newsplit配置
"""
_base_ = [
    './nusc_baseline_480_60x30_30e.py'
]

# 修改head配置，启用streaming和速度先验
model = dict(
    head_cfg=dict(
        streaming_cfg=dict(
            streaming=True,           # 启用streaming
            batch_size=1,
            topk=100,                 # baseline用100个query
            trans_loss_weight=5.0,
            use_velocity_prior=True,  # 🆕 启用速度先验
        ),
    ),
)

# 注意：这个配置同时引入了两个改动：
# 1. 启用streaming机制（原baseline没有）
# 2. 启用速度先验（我们的改进）
# 
# 如果要做严格的消融实验，建议对比：
# - nusc_baseline_480_60x30_30e.py (无streaming, 无速度)
# - nusc_baseline_480_60x30_30e_streaming.py (有streaming, 无速度) 
# - nusc_baseline_480_60x30_30e_velocity.py (有streaming, 有速度) ← 本配置


