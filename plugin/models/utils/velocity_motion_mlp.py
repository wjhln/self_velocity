"""
带速度先验的Query更新模块
"""
import torch
import torch.nn as nn


class VelocityMotionMLP(nn.Module):
    """
    融合位姿变换和速度信息的Query更新模块
    
    Args:
        pose_dim (int): 位姿编码维度 (默认12: 3x3旋转矩阵+3平移向量)
        velocity_dim (int): 速度编码维度 (默认4: vx, vy, |v|, dt)
        f_dim (int): query特征维度
        use_velocity (bool): 是否使用速度信息
        identity (bool): 是否使用残差连接
    """
    
    def __init__(self, pose_dim=12, velocity_dim=4, f_dim=512, 
                 use_velocity=True, identity=True):
        super().__init__()
        self.pose_dim = pose_dim
        self.velocity_dim = velocity_dim
        self.f_dim = f_dim
        self.use_velocity = use_velocity
        self.identity = identity
        
        # 计算输入维度
        if use_velocity:
            c_dim = pose_dim + velocity_dim
        else:
            c_dim = pose_dim
        
        self.c_dim = c_dim
        
        # MLP网络
        self.fc = nn.Sequential(
            nn.Linear(c_dim + f_dim, 2*f_dim),
            nn.LayerNorm(2*f_dim),
            nn.ReLU(),
            nn.Linear(2*f_dim, f_dim)
        )
        
        # 速度置信度网络（可选）
        if use_velocity:
            self.confidence_net = nn.Sequential(
                nn.Linear(velocity_dim, 32),
                nn.ReLU(),
                nn.Linear(32, 1),
                nn.Sigmoid()
            )
        
        self.init_weights()
    
    def init_weights(self):
        """初始化权重"""
        for m in self.fc:
            if hasattr(m, 'weight') and m.weight is not None:
                if m.weight.dim() > 1:
                    if self.identity:
                        nn.init.zeros_(m.weight)
                    else:
                        nn.init.xavier_uniform_(m.weight)
            if hasattr(m, 'bias') and m.bias is not None:
                nn.init.zeros_(m.bias)
        
        # 置信度网络初始化为输出0.5
        if self.use_velocity:
            for m in self.confidence_net:
                if hasattr(m, 'weight') and m.weight is not None:
                    if m.weight.dim() > 1:
                        nn.init.xavier_uniform_(m.weight)
                if hasattr(m, 'bias') and m.bias is not None:
                    nn.init.zeros_(m.bias)
    
    def forward(self, x, pose_encoding, velocity_encoding=None):
        """
        Args:
            x: query特征 (N, f_dim)
            pose_encoding: 位姿编码 (N, pose_dim) 或 (1, pose_dim)
            velocity_encoding: 速度编码 (N, velocity_dim) 或 (1, velocity_dim) 或 None
        
        Returns:
            out: 更新后的query特征 (N, f_dim)
        """
        # 拼接编码
        if self.use_velocity and velocity_encoding is not None:
            # 广播到相同shape
            if pose_encoding.dim() == 2 and pose_encoding.size(0) == 1:
                pose_encoding = pose_encoding.repeat(x.size(0), 1)
            if velocity_encoding.dim() == 2 and velocity_encoding.size(0) == 1:
                velocity_encoding = velocity_encoding.repeat(x.size(0), 1)
            
            c = torch.cat([pose_encoding, velocity_encoding], dim=-1)
            
            # 计算速度置信度（可选）
            # confidence = self.confidence_net(velocity_encoding)  # (N, 1)
        else:
            if pose_encoding.dim() == 2 and pose_encoding.size(0) == 1:
                pose_encoding = pose_encoding.repeat(x.size(0), 1)
            c = pose_encoding
        
        # MLP更新
        xc = torch.cat([x, c], dim=-1)
        out = self.fc(xc)
        
        # 残差连接
        if self.identity:
            out = out + x
        
        return out


class AdaptiveVelocityMotionMLP(nn.Module):
    """
    自适应融合位姿和速度的Query更新模块
    根据速度大小和加速度自动调整融合权重
    
    Args:
        pose_dim (int): 位姿编码维度
        velocity_dim (int): 速度编码维度  
        f_dim (int): query特征维度
        identity (bool): 是否使用残差连接
    """
    
    def __init__(self, pose_dim=12, velocity_dim=4, f_dim=512, identity=True):
        super().__init__()
        self.pose_dim = pose_dim
        self.velocity_dim = velocity_dim
        self.f_dim = f_dim
        self.identity = identity
        
        # 位姿分支
        self.pose_branch = nn.Sequential(
            nn.Linear(pose_dim + f_dim, 2*f_dim),
            nn.LayerNorm(2*f_dim),
            nn.ReLU(),
            nn.Linear(2*f_dim, f_dim)
        )
        
        # 速度分支
        self.velocity_branch = nn.Sequential(
            nn.Linear(velocity_dim + f_dim, 2*f_dim),
            nn.LayerNorm(2*f_dim),
            nn.ReLU(),
            nn.Linear(2*f_dim, f_dim)
        )
        
        # 自适应权重网络
        self.weight_net = nn.Sequential(
            nn.Linear(velocity_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 1),
            nn.Sigmoid()
        )
        
        self.init_weights()
    
    def init_weights(self):
        """初始化权重"""
        for branch in [self.pose_branch, self.velocity_branch]:
            for m in branch:
                if hasattr(m, 'weight') and m.weight is not None:
                    if m.weight.dim() > 1:
                        if self.identity:
                            nn.init.zeros_(m.weight)
                        else:
                            nn.init.xavier_uniform_(m.weight)
                if hasattr(m, 'bias') and m.bias is not None:
                    nn.init.zeros_(m.bias)
        
        # 权重网络初始化为输出0.5
        for m in self.weight_net:
            if hasattr(m, 'weight') and m.weight is not None:
                if m.weight.dim() > 1:
                    nn.init.xavier_uniform_(m.weight)
            if hasattr(m, 'bias') and m.bias is not None:
                nn.init.constant_(m.bias, 0.0)
    
    def forward(self, x, pose_encoding, velocity_encoding):
        """
        Args:
            x: query特征 (N, f_dim)
            pose_encoding: 位姿编码 (N, pose_dim) 或 (1, pose_dim)
            velocity_encoding: 速度编码 (N, velocity_dim) 或 (1, velocity_dim)
        
        Returns:
            out: 更新后的query特征 (N, f_dim)
        """
        # 广播
        if pose_encoding.dim() == 2 and pose_encoding.size(0) == 1:
            pose_encoding = pose_encoding.repeat(x.size(0), 1)
        if velocity_encoding.dim() == 2 and velocity_encoding.size(0) == 1:
            velocity_encoding = velocity_encoding.repeat(x.size(0), 1)
        
        # 位姿分支
        xp = torch.cat([x, pose_encoding], dim=-1)
        out_pose = self.pose_branch(xp)
        
        # 速度分支
        xv = torch.cat([x, velocity_encoding], dim=-1)
        out_velocity = self.velocity_branch(xv)
        
        # 自适应权重
        weight = self.weight_net(velocity_encoding)  # (N, 1)
        
        # 融合
        out = weight * out_velocity + (1 - weight) * out_pose
        
        # 残差连接
        if self.identity:
            out = out + x
        
        return out

