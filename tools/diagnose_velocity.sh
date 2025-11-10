#!/bin/bash
# 速度先验诊断工具

echo "🔍 速度先验功能诊断"
echo "===================="

cd /home/wang/Project/Perception/StreamMapNet

echo -e "\n📋 步骤1: 检查数据加载..."
echo "----------------------------"
python tools/test_velocity_prior.py 2>&1 | grep -E "✅|❌|测试" | head -20

echo -e "\n📋 步骤2: 检查速度计算准确性..."
echo "----------------------------"
python tools/verify_velocity_coordinate.py --num-samples 100 2>&1 | tail -30

echo -e "\n📋 步骤3: 检查配置文件..."
echo "----------------------------"
if grep -q "use_velocity_prior.*True" plugin/configs/nusc_newsplit_480_60x30_24e_velocity.py; then
    echo "✅ 配置文件中已启用速度先验"
else
    echo "❌ 配置文件中未启用速度先验！"
fi

echo -e "\n📋 步骤4: 检查模块导入..."
echo "----------------------------"
python << 'EOF'
try:
    from plugin.models.utils.velocity_motion_mlp import VelocityMotionMLP
    print("✅ VelocityMotionMLP 导入成功")
except Exception as e:
    print(f"❌ VelocityMotionMLP 导入失败: {e}")

try:
    from plugin.datasets.nusc_dataset import NuscDataset
    print("✅ NuscDataset 导入成功")
except Exception as e:
    print(f"❌ NuscDataset 导入失败: {e}")
EOF

echo -e "\n📋 步骤5: 检查训练日志（如果存在）..."
echo "----------------------------"
if [ -f "work_dirs/nusc_newsplit_480_60x30_24e_velocity/log.txt" ]; then
    echo "查找速度相关日志..."
    grep -i "velocity" work_dirs/nusc_newsplit_480_60x30_24e_velocity/log.txt | head -5
    if [ $? -eq 0 ]; then
        echo "✅ 找到速度相关日志"
    else
        echo "⚠️  未找到速度相关日志（可能未添加调试输出）"
    fi
else
    echo "⚠️  训练日志不存在，请先训练模型"
fi

echo -e "\n" 
echo "===================="
echo "诊断完成！"
echo "===================="
echo -e "\n如果发现问题，请查看 DEBUG_NO_IMPROVEMENT.md 获取解决方案"

