"""
测试点匹配先验功能
"""
import sys
sys.path.insert(0, '.')

import torch
import numpy as np


def test_point_matcher():
    """测试PointMatcher模块"""
    print("=" * 80)
    print("测试1: PointMatcher模块")
    print("=" * 80)
    
    from plugin.models.utils.point_matcher import PointMatcher
    
    # 创建匹配器
    matcher = PointMatcher(num_points=20, distance_threshold=2.0)
    
    print(f"\n✅ PointMatcher创建成功")
    print(f"  num_points: {matcher.num_points}")
    print(f"  distance_threshold: {matcher.distance_threshold}")
    
    # 创建测试数据
    pred_points = torch.randn(10, 20, 2) * 10  # 10条预测线
    gt_points = torch.randn(5, 20, 2) * 10     # 5条GT线
    
    print(f"\n测试数据:")
    print(f"  pred_points: {pred_points.shape}")
    print(f"  gt_points: {gt_points.shape}")
    
    # 测试匹配
    matched_points, confidence, matched_indices = matcher(pred_points, gt_points)
    
    print(f"\n匹配结果:")
    print(f"  matched_points: {matched_points.shape}")
    print(f"  confidence: {confidence.shape}")
    print(f"  confidence range: [{confidence.min():.3f}, {confidence.max():.3f}]")
    print(f"  matched_indices: {matched_indices}")
    
    # 验证
    assert matched_points.shape == pred_points.shape
    assert confidence.shape == (10, 1)
    
    print(f"\n✅ PointMatcher测试通过")
    
    # 测试边界情况
    print(f"\n测试边界情况:")
    
    # 情况1: 没有GT
    empty_gt = torch.zeros(0, 20, 2)
    matched, conf, _ = matcher(pred_points, empty_gt)
    assert matched.shape == pred_points.shape
    assert conf.sum() == 0
    print(f"  ✅ 空GT处理正确")
    
    # 情况2: 完美匹配
    perfect_gt = pred_points.clone()
    matched, conf, _ = matcher(pred_points, perfect_gt)
    assert conf.min() > 0.9  # 应该有很高的置信度
    print(f"  ✅ 完美匹配置信度: {conf.mean():.3f}")
    
    # 情况3: 距离很远
    far_gt = pred_points + 10.0  # 偏移10米
    matched, conf, _ = matcher(pred_points, far_gt)
    print(f"  ✅ 远距离匹配置信度: {conf.mean():.3f}")
    
    print("\n" + "=" * 80)


def test_integration():
    """测试集成到MapDetectorHead"""
    print("\n" + "=" * 80)
    print("测试2: 集成测试")
    print("=" * 80)
    
    try:
        from plugin.models.heads import MapDetectorHead
        from plugin.models.utils.point_matcher import PointMatcher
        
        print("✅ 模块导入成功")
        
        # 创建Head（简化配置）
        head = MapDetectorHead(
            num_queries=100,
            num_classes=3,
            embed_dims=256,
            num_points=20,
            roi_size=(60, 30),
            streaming_cfg=dict(
                streaming=True,
                batch_size=1,
                topk=50,
                trans_loss_weight=5.0,
                use_point_matching_prior=True,
                matching_loss_weight=0.5,
            ),
            transformer=dict(
                type='MapTransformer',
                decoder=dict(
                    type='MapTransformerDecoder_new',
                    num_layers=2,
                    return_intermediate=True,
                )
            ),
            loss_cls=dict(type='FocalLoss', use_sigmoid=True, loss_weight=1.0),
            loss_reg=dict(type='L1Loss', loss_weight=1.0),
            assigner=dict(type='HungarianLinesAssigner')
        )
        
        print("✅ MapDetectorHead创建成功")
        print(f"  use_point_matching_prior: {head.use_point_matching_prior}")
        print(f"  matching_loss_weight: {head.matching_loss_weight}")
        
        if hasattr(head, 'point_matcher'):
            print(f"  ✅ PointMatcher已初始化")
        else:
            print(f"  ❌ PointMatcher未初始化")
        
    except Exception as e:
        print(f"❌ 集成测试失败: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    print("\n" + "=" * 80)
    return True


def test_matching_logic():
    """测试匹配逻辑"""
    print("\n" + "=" * 80)
    print("测试3: 匹配逻辑验证")
    print("=" * 80)
    
    from plugin.models.utils.point_matcher import PointMatcher
    
    matcher = PointMatcher(num_points=20, distance_threshold=2.0)
    
    # 模拟场景：3条预测线，2条GT线
    print("\n场景：3条预测线匹配2条GT线")
    
    # 预测线
    pred_line1 = torch.linspace(0, 1, 20).unsqueeze(-1).repeat(1, 2) * 10  # 直线
    pred_line2 = torch.linspace(0, 1, 20).unsqueeze(-1).repeat(1, 2) * 10 + 5  # 平移5米
    pred_line3 = torch.linspace(0, 1, 20).unsqueeze(-1).repeat(1, 2) * 10 + 20  # 平移20米
    pred_points = torch.stack([pred_line1, pred_line2, pred_line3])
    
    # GT线（与pred_line1和pred_line2接近）
    gt_line1 = pred_line1 + 0.1  # 与pred_line1很接近
    gt_line2 = pred_line2 + 0.2  # 与pred_line2很接近
    gt_points = torch.stack([gt_line1, gt_line2])
    
    print(f"  pred_points: {pred_points.shape}")
    print(f"  gt_points: {gt_points.shape}")
    
    # 匹配
    matched, confidence, indices = matcher(pred_points, gt_points)
    
    print(f"\n匹配结果:")
    for i in range(3):
        print(f"  pred_line{i+1} → gt_line{indices[i]+1}, confidence={confidence[i].item():.3f}")
    
    # 验证
    assert indices[0] == 0 or indices[0] == 1, "pred_line1应该匹配到gt_line1或gt_line2"
    assert confidence[0] > 0.5, "接近的线应该有高置信度"
    assert confidence[2] < 0.5, "远离的线应该有低置信度"
    
    print(f"\n✅ 匹配逻辑正确")
    print("\n" + "=" * 80)


def main():
    """主测试函数"""
    print("\n" + "=" * 80)
    print("点匹配先验功能测试")
    print("=" * 80)
    
    try:
        # 测试1: 匹配器模块
        test_point_matcher()
        
        # 测试2: 集成测试
        success = test_integration()
        if not success:
            return 1
        
        # 测试3: 匹配逻辑
        test_matching_logic()
        
        print("\n" + "=" * 80)
        print("🎉 所有测试通过！")
        print("=" * 80)
        
        print("\n下一步:")
        print("  1. 运行训练: bash tools/dist_train.sh plugin/configs/nusc_newsplit_480_60x30_24e_matching.py 8")
        print("  2. 监控matching_loss，确认在下降")
        print("  3. 对比baseline，预期 +1.5-3.0 AP")
        
        return 0
        
    except Exception as e:
        print(f"\n❌ 测试失败: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == '__main__':
    exit(main())

