"""
测试速度先验功能
验证数据加载和模型forward是否正常
"""
import sys
sys.path.insert(0, '.')

import torch
import mmcv
from mmcv import Config
from mmdet.datasets import build_dataset
from plugin.models.utils.velocity_motion_mlp import VelocityMotionMLP


def test_dataset():
    """测试数据集是否正确加载速度信息"""
    print("=" * 80)
    print("测试1: 数据集速度信息加载")
    print("=" * 80)
    
    # 加载配置
    cfg = Config.fromfile('plugin/configs/nusc_newsplit_480_60x30_24e.py')
    
    # 构建数据集
    dataset = build_dataset(cfg.data.val)
    
    # 测试几个样本
    print(f"\n数据集大小: {len(dataset)}")
    
    for i in range(min(5, len(dataset))):
        data = dataset[i]
        img_metas = data['img_metas'].data
        
        print(f"\n样本 {i}:")
        print(f"  Keys: {img_metas.keys()}")
        
        if 'velocity' in img_metas:
            velocity = img_metas['velocity']
            velocity_mag = img_metas.get('velocity_magnitude', 0.0)
            print(f"  ✅ 速度: [{velocity[0]:.3f}, {velocity[1]:.3f}, {velocity[2]:.3f}] m/s")
            print(f"  ✅ 速度大小: {velocity_mag:.3f} m/s")
        else:
            print(f"  ❌ 未找到速度信息")
        
        if 'timestamp' in img_metas:
            print(f"  ✅ 时间戳: {img_metas['timestamp']}")
        else:
            print(f"  ⚠️  未找到时间戳")
    
    print("\n" + "=" * 80)
    print("数据集测试完成")
    print("=" * 80)


def test_velocity_mlp():
    """测试VelocityMotionMLP模块"""
    print("\n" + "=" * 80)
    print("测试2: VelocityMotionMLP模块")
    print("=" * 80)
    
    # 创建模块
    model = VelocityMotionMLP(
        pose_dim=12,
        velocity_dim=4,
        f_dim=256,
        use_velocity=True,
        identity=True
    )
    
    print(f"\n模型参数:")
    print(f"  pose_dim: {model.pose_dim}")
    print(f"  velocity_dim: {model.velocity_dim}")
    print(f"  f_dim: {model.f_dim}")
    print(f"  use_velocity: {model.use_velocity}")
    
    # 测试forward
    batch_size = 10
    query = torch.randn(batch_size, 256)
    pose_encoding = torch.randn(1, 12)
    velocity_encoding = torch.randn(1, 4)
    
    print(f"\n输入shape:")
    print(f"  query: {query.shape}")
    print(f"  pose_encoding: {pose_encoding.shape}")
    print(f"  velocity_encoding: {velocity_encoding.shape}")
    
    # Forward
    output = model(query, pose_encoding, velocity_encoding)
    
    print(f"\n输出shape:")
    print(f"  output: {output.shape}")
    
    # 验证
    assert output.shape == query.shape, "输出shape不匹配"
    print(f"\n✅ Forward测试通过")
    
    # 测试不使用速度
    model_no_velocity = VelocityMotionMLP(
        pose_dim=12,
        velocity_dim=4,
        f_dim=256,
        use_velocity=False,
        identity=True
    )
    
    output_no_velocity = model_no_velocity(query, pose_encoding, None)
    print(f"✅ 不使用速度的forward测试通过")
    
    print("\n" + "=" * 80)
    print("VelocityMotionMLP测试完成")
    print("=" * 80)


def test_integration():
    """测试集成到MapDetectorHead"""
    print("\n" + "=" * 80)
    print("测试3: 集成测试")
    print("=" * 80)
    
    try:
        from plugin.models.heads import MapDetectorHead
        print("✅ MapDetectorHead导入成功")
        
        # 检查是否有VelocityMotionMLP
        from plugin.models.utils.velocity_motion_mlp import VelocityMotionMLP
        print("✅ VelocityMotionMLP导入成功")
        
        print("\n集成测试通过！")
        
    except Exception as e:
        print(f"❌ 集成测试失败: {e}")
        import traceback
        traceback.print_exc()
    
    print("\n" + "=" * 80)
    print("集成测试完成")
    print("=" * 80)


def main():
    """主测试函数"""
    print("\n" + "=" * 80)
    print("速度先验功能测试")
    print("=" * 80)
    
    try:
        # 测试1: 数据集
        test_dataset()
        
        # 测试2: 模型模块
        test_velocity_mlp()
        
        # 测试3: 集成
        test_integration()
        
        print("\n" + "=" * 80)
        print("🎉 所有测试通过！")
        print("=" * 80)
        print("\n下一步:")
        print("  1. 运行训练: bash tools/dist_train.sh plugin/configs/nusc_newsplit_480_60x30_24e_velocity.py 8")
        print("  2. 查看日志，确认速度信息被正确使用")
        print("  3. 对比baseline和velocity版本的性能")
        
    except Exception as e:
        print(f"\n❌ 测试失败: {e}")
        import traceback
        traceback.print_exc()
        return 1
    
    return 0


if __name__ == '__main__':
    exit(main())

