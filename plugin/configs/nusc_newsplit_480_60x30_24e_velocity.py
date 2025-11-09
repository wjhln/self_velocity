"""
带速度先验的StreamMapNet配置
基于 nusc_newsplit_480_60x30_24e.py
"""
_base_ = [
    './nusc_newsplit_480_60x30_24e.py'
]

# 修改streaming配置，启用速度先验
model = dict(
    pts_bbox_head=dict(
        streaming_cfg=dict(
            streaming=True,
            batch_size=1,
            topk=300,
            trans_loss_weight=5.0,
            use_velocity_prior=True,  # 🆕 启用速度先验
        ),
    ),
)

# 可以在这里调整学习率等超参数
# optimizer = dict(
#     type='AdamW',
#     lr=6e-4,  # 可能需要稍微调整
#     weight_decay=0.01,
# )

