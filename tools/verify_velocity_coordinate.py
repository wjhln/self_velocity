"""
验证can_bus速度的坐标系
对比速度积分与位姿变化，确认坐标系是否一致
"""
import pickle
import numpy as np
import argparse
from pyquaternion import Quaternion
import matplotlib.pyplot as plt


def compute_displacement_from_pose(sample1, sample2):
    """从位姿计算位移（在sample2的ego坐标系下）"""
    # 上一帧位姿
    prev_e2g_trans = np.array(sample1['ego2global_translation'])
    prev_e2g_rot = Quaternion(sample1['ego2global_rotation']).rotation_matrix
    
    # 当前帧位姿
    curr_e2g_trans = np.array(sample2['ego2global_translation'])
    curr_e2g_rot = Quaternion(sample2['ego2global_rotation']).rotation_matrix
    
    # 构建变换矩阵
    prev_e2g_matrix = np.eye(4)
    prev_e2g_matrix[:3, :3] = prev_e2g_rot
    prev_e2g_matrix[:3, 3] = prev_e2g_trans
    
    curr_g2e_matrix = np.eye(4)
    curr_g2e_matrix[:3, :3] = curr_e2g_rot.T
    curr_g2e_matrix[:3, 3] = -(curr_e2g_rot.T @ curr_e2g_trans)
    
    # prev -> curr 变换
    prev2curr_matrix = curr_g2e_matrix @ prev_e2g_matrix
    
    # 提取位移
    displacement = prev2curr_matrix[:3, 3]
    
    return displacement, prev2curr_matrix


def compute_displacement_from_velocity(sample1, sample2):
    """从速度积分计算位移"""
    if 'can_bus' not in sample1:
        return None, None
    
    # 提取速度 (索引7-9)
    can_bus = sample1['can_bus']
    velocity = np.array([can_bus[7], can_bus[8], can_bus[9]])
    
    # 计算时间差
    dt = (sample2['timestamp'] - sample1['timestamp']) / 1e6  # 微秒转秒
    
    # 速度积分
    displacement = velocity * dt
    
    return displacement, velocity


def angle_between_vectors(v1, v2):
    """计算两个向量的夹角（度）"""
    v1_norm = v1 / (np.linalg.norm(v1) + 1e-8)
    v2_norm = v2 / (np.linalg.norm(v2) + 1e-8)
    cos_angle = np.clip(np.dot(v1_norm, v2_norm), -1.0, 1.0)
    angle = np.degrees(np.arccos(cos_angle))
    return angle


def verify_coordinate_system(data_path, num_samples=100):
    """验证坐标系"""
    print("=" * 80)
    print("验证can_bus速度的坐标系")
    print("=" * 80)
    
    # 加载数据
    print(f"\n加载数据: {data_path}")
    data = pickle.load(open(data_path, 'rb'))
    
    if isinstance(data, dict) and 'infos' in data:
        samples = data['infos']
    else:
        samples = data
    
    print(f"总样本数: {len(samples)}")
    
    # 统计信息
    results = {
        'angle_diffs': [],
        'magnitude_ratios': [],
        'pose_displacements': [],
        'velocity_displacements': [],
        'velocities': [],
        'dt_list': []
    }
    
    valid_count = 0
    
    # 遍历相邻帧
    for i in range(min(num_samples, len(samples) - 1)):
        sample1 = samples[i]
        sample2 = samples[i + 1]
        
        # 跳过不同场景
        if sample1.get('scene_name') != sample2.get('scene_name'):
            continue
        
        # 从位姿计算位移
        disp_pose, _ = compute_displacement_from_pose(sample1, sample2)
        
        # 从速度计算位移
        disp_velocity, velocity = compute_displacement_from_velocity(sample1, sample2)
        
        if disp_velocity is None:
            continue
        
        # 计算时间差
        dt = (sample2['timestamp'] - sample1['timestamp']) / 1e6
        
        # 只考虑运动帧
        velocity_magnitude = np.linalg.norm(velocity[:2])
        if velocity_magnitude < 0.1:  # 静止
            continue
        
        # 只考虑合理的时间间隔
        if dt < 0.1 or dt > 2.0:
            continue
        
        # 计算角度差（只看xy平面）
        angle_diff = angle_between_vectors(disp_pose[:2], disp_velocity[:2])
        
        # 计算大小比例
        magnitude_pose = np.linalg.norm(disp_pose[:2])
        magnitude_velocity = np.linalg.norm(disp_velocity[:2])
        
        if magnitude_velocity > 0.01:
            magnitude_ratio = magnitude_pose / magnitude_velocity
        else:
            continue
        
        # 记录
        results['angle_diffs'].append(angle_diff)
        results['magnitude_ratios'].append(magnitude_ratio)
        results['pose_displacements'].append(disp_pose)
        results['velocity_displacements'].append(disp_velocity)
        results['velocities'].append(velocity)
        results['dt_list'].append(dt)
        
        valid_count += 1
        
        # 打印前几个样本
        if valid_count <= 5:
            print(f"\n样本 {i}:")
            print(f"  时间间隔: {dt:.3f}s")
            print(f"  速度: [{velocity[0]:.2f}, {velocity[1]:.2f}, {velocity[2]:.2f}] m/s")
            print(f"  位姿位移: [{disp_pose[0]:.3f}, {disp_pose[1]:.3f}, {disp_pose[2]:.3f}] m")
            print(f"  速度位移: [{disp_velocity[0]:.3f}, {disp_velocity[1]:.3f}, {disp_velocity[2]:.3f}] m")
            print(f"  角度差: {angle_diff:.2f}°")
            print(f"  大小比例: {magnitude_ratio:.3f}")
    
    # 统计分析
    print("\n" + "=" * 80)
    print(f"统计分析 (有效样本数: {valid_count})")
    print("=" * 80)
    
    angle_diffs = np.array(results['angle_diffs'])
    magnitude_ratios = np.array(results['magnitude_ratios'])
    
    print(f"\n角度差统计:")
    print(f"  平均值: {angle_diffs.mean():.2f}°")
    print(f"  中位数: {np.median(angle_diffs):.2f}°")
    print(f"  标准差: {angle_diffs.std():.2f}°")
    print(f"  最小值: {angle_diffs.min():.2f}°")
    print(f"  最大值: {angle_diffs.max():.2f}°")
    
    print(f"\n大小比例统计:")
    print(f"  平均值: {magnitude_ratios.mean():.3f}")
    print(f"  中位数: {np.median(magnitude_ratios):.3f}")
    print(f"  标准差: {magnitude_ratios.std():.3f}")
    print(f"  最小值: {magnitude_ratios.min():.3f}")
    print(f"  最大值: {magnitude_ratios.max():.3f}")
    
    # 判断结果
    print("\n" + "=" * 80)
    print("结论:")
    print("=" * 80)
    
    median_angle = np.median(angle_diffs)
    median_ratio = np.median(magnitude_ratios)
    
    if median_angle < 10 and 0.8 < median_ratio < 1.2:
        print("✅ 坐标系一致！can_bus速度可以直接使用")
        print(f"   - 角度差中位数: {median_angle:.2f}° (< 10°)")
        print(f"   - 大小比例中位数: {median_ratio:.3f} (0.8-1.2)")
        coordinate_ok = True
    elif median_angle > 170 and 0.8 < median_ratio < 1.2:
        print("⚠️  速度方向相反！需要取负号")
        print(f"   - 角度差中位数: {median_angle:.2f}° (接近180°)")
        print(f"   - 建议: velocity = -can_bus[7:10]")
        coordinate_ok = True
    elif median_angle < 10 and median_ratio > 1.5:
        print("⚠️  速度单位可能不对！")
        print(f"   - 大小比例中位数: {median_ratio:.3f}")
        print(f"   - 可能需要缩放因子: {median_ratio:.3f}")
        coordinate_ok = False
    else:
        print("❌ 坐标系不一致！需要坐标变换")
        print(f"   - 角度差中位数: {median_angle:.2f}°")
        print(f"   - 大小比例中位数: {median_ratio:.3f}")
        print("   - 建议: 检查数据或进行坐标变换")
        coordinate_ok = False
    
    # 可视化
    if valid_count > 0:
        plot_results(results)
    
    return coordinate_ok, results


def plot_results(results):
    """可视化结果"""
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    
    # 角度差分布
    axes[0, 0].hist(results['angle_diffs'], bins=50, edgecolor='black')
    axes[0, 0].set_xlabel('Angle Difference (degrees)')
    axes[0, 0].set_ylabel('Frequency')
    axes[0, 0].set_title('Angle Difference Distribution')
    axes[0, 0].axvline(np.median(results['angle_diffs']), color='r', 
                       linestyle='--', label=f'Median: {np.median(results["angle_diffs"]):.2f}°')
    axes[0, 0].legend()
    
    # 大小比例分布
    axes[0, 1].hist(results['magnitude_ratios'], bins=50, edgecolor='black')
    axes[0, 1].set_xlabel('Magnitude Ratio (pose/velocity)')
    axes[0, 1].set_ylabel('Frequency')
    axes[0, 1].set_title('Magnitude Ratio Distribution')
    axes[0, 1].axvline(np.median(results['magnitude_ratios']), color='r', 
                       linestyle='--', label=f'Median: {np.median(results["magnitude_ratios"]):.3f}')
    axes[0, 1].legend()
    
    # 位移对比 (x方向)
    pose_x = [d[0] for d in results['pose_displacements']]
    velocity_x = [d[0] for d in results['velocity_displacements']]
    axes[1, 0].scatter(velocity_x, pose_x, alpha=0.5, s=10)
    axes[1, 0].plot([min(velocity_x), max(velocity_x)], 
                    [min(velocity_x), max(velocity_x)], 'r--', label='y=x')
    axes[1, 0].set_xlabel('Velocity Displacement X (m)')
    axes[1, 0].set_ylabel('Pose Displacement X (m)')
    axes[1, 0].set_title('X Direction Comparison')
    axes[1, 0].legend()
    axes[1, 0].grid(True, alpha=0.3)
    
    # 位移对比 (y方向)
    pose_y = [d[1] for d in results['pose_displacements']]
    velocity_y = [d[1] for d in results['velocity_displacements']]
    axes[1, 1].scatter(velocity_y, pose_y, alpha=0.5, s=10)
    axes[1, 1].plot([min(velocity_y), max(velocity_y)], 
                    [min(velocity_y), max(velocity_y)], 'r--', label='y=x')
    axes[1, 1].set_xlabel('Velocity Displacement Y (m)')
    axes[1, 1].set_ylabel('Pose Displacement Y (m)')
    axes[1, 1].set_title('Y Direction Comparison')
    axes[1, 1].legend()
    axes[1, 1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('velocity_coordinate_verification.png', dpi=150)
    print(f"\n可视化结果已保存: velocity_coordinate_verification.png")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--data-path', type=str, 
                       default='datasets/nuScenes/nuscenes_infos_temporal_val.pkl',
                       help='Path to the data file')
    parser.add_argument('--num-samples', type=int, default=200,
                       help='Number of samples to verify')
    
    args = parser.parse_args()
    
    coordinate_ok, results = verify_coordinate_system(args.data_path, args.num_samples)

