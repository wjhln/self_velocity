"""
带点匹配先验的StreamMapNet配置
基于 nusc_newsplit_480_60x30_24e.py

核心改进：使用点对点匹配作为先验，而不是速度信息
预期提升：+1.5-3.0 AP
"""
_base_ = [
    './nusc_newsplit_480_60x30_24e.py'
]

# 修改streaming配置，启用点匹配先验
model = dict(
    head_cfg=dict(
        streaming_cfg=dict(
            streaming=True,
            # batch_size 会从基础配置继承，不要覆盖
            # topk 保持基础配置的 33 (num_queries * 1/3)
            trans_loss_weight=5.0,
            use_velocity_prior=False,          # 不使用速度先验
            use_point_matching_prior=True,     # 🆕 启用点匹配先验
            matching_loss_weight=0.5,          # 🆕 匹配loss权重
        ),
    ),
)

# 可以调整的超参数
# matching_loss_weight: 0.3-1.0 (匹配loss的权重)
#   - 0.3: 弱约束，主要靠几何变换
#   - 0.5: 平衡（推荐）
#   - 1.0: 强约束，更依赖匹配

