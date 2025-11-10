"""
点对点匹配模块
用于建立上一帧传播点与当前帧GT点的对应关系
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


class PointMatcher(nn.Module):
    """
    简单的点匹配器，使用最近邻匹配
    
    Args:
        num_points (int): 每条线的点数
        distance_threshold (float): 匹配距离阈值（米）
        confidence_sigma (float): 置信度计算的sigma参数
    """
    
    def __init__(self, num_points=20, distance_threshold=2.0, confidence_sigma=1.0):
        super().__init__()
        self.num_points = num_points
        self.distance_threshold = distance_threshold
        self.confidence_sigma = confidence_sigma
    
    def compute_chamfer_distance(self, points1, points2):
        """
        计算两组点之间的Chamfer距离
        
        Args:
            points1: (N, num_points, 2)
            points2: (M, num_points, 2)
        
        Returns:
            distances: (N, M) 距离矩阵
        """
        N = points1.shape[0]
        M = points2.shape[0]
        
        # 扩展维度用于广播
        p1 = points1.unsqueeze(1)  # (N, 1, num_points, 2)
        p2 = points2.unsqueeze(0)  # (1, M, num_points, 2)
        
        # 计算点到点的距离
        point_distances = (p1 - p2).pow(2).sum(-1).sqrt()  # (N, M, num_points)
        
        # Chamfer距离：双向最近邻的平均
        # 方向1: p1中每个点到p2的最近距离
        dist_1to2 = point_distances.min(dim=2)[0].mean(dim=2)  # (N, M)
        # 方向2: p2中每个点到p1的最近距离  
        dist_2to1 = point_distances.min(dim=1)[0].mean(dim=2)  # (N, M)
        
        # 对称Chamfer距离
        distances = (dist_1to2 + dist_2to1) / 2.0
        
        return distances
    
    def compute_simple_distance(self, points1, points2):
        """
        简单的平均距离（更快）
        
        Args:
            points1: (N, num_points, 2)
            points2: (M, num_points, 2)
        
        Returns:
            distances: (N, M)
        """
        N = points1.shape[0]
        M = points2.shape[0]
        
        # 扩展维度
        p1 = points1.unsqueeze(1)  # (N, 1, num_points, 2)
        p2 = points2.unsqueeze(0)  # (1, M, num_points, 2)
        
        # 对应点的平均距离
        distances = (p1 - p2).pow(2).sum(-1).sqrt().mean(-1)  # (N, M)
        
        return distances
    
    def forward(self, pred_points, gt_points, use_chamfer=False):
        """
        将预测点匹配到GT点
        
        Args:
            pred_points: (N, num_points, 2) 传播的参考点
            gt_points: (M, num_points, 2) GT点
        
        Returns:
            matched_points: (N, num_points, 2) 匹配的GT点
            confidence: (N, 1) 匹配置信度 [0, 1]
            matched_indices: (N,) 匹配的GT索引
        """
        N = pred_points.shape[0]
        M = gt_points.shape[0]
        
        if M == 0:
            # 没有GT，返回原始点，置信度为0
            return pred_points, torch.zeros(N, 1, device=pred_points.device), None
        
        # 计算距离矩阵
        if use_chamfer:
            distances = self.compute_chamfer_distance(pred_points, gt_points)
        else:
            distances = self.compute_simple_distance(pred_points, gt_points)
        
        # 最近邻匹配
        min_distances, matched_indices = distances.min(dim=1)  # (N,)
        
        # 获取匹配的GT点
        matched_points = gt_points[matched_indices]  # (N, num_points, 2)
        
        # 计算置信度：距离越小，置信度越高
        # confidence = exp(-distance / sigma)
        confidence = torch.exp(-min_distances / self.confidence_sigma)  # (N,)
        confidence = confidence.unsqueeze(-1)  # (N, 1)
        
        # 距离太远的，置信度设为0
        confidence[min_distances > self.distance_threshold] = 0.0
        
        return matched_points, confidence, matched_indices


class AdaptivePointMatcher(nn.Module):
    """
    自适应点匹配器，根据场景动态调整匹配策略
    
    Args:
        num_points (int): 每条线的点数
        embed_dim (int): 特征维度
    """
    
    def __init__(self, num_points=20, embed_dim=128):
        super().__init__()
        self.num_points = num_points
        self.embed_dim = embed_dim
        
        # 点特征编码器
        self.point_encoder = nn.Sequential(
            nn.Linear(2, 64),
            nn.ReLU(),
            nn.Linear(64, embed_dim)
        )
        
        # 匹配得分网络
        self.matching_net = nn.Sequential(
            nn.Linear(embed_dim * 2, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, 1)
        )
        
        # 置信度网络
        self.confidence_net = nn.Sequential(
            nn.Linear(embed_dim * 2, 64),
            nn.ReLU(),
            nn.Linear(64, 1),
            nn.Sigmoid()
        )
    
    def encode_points(self, points):
        """
        编码点集为特征向量
        
        Args:
            points: (N, num_points, 2)
        
        Returns:
            features: (N, embed_dim)
        """
        # 对每个点编码
        point_features = self.point_encoder(points)  # (N, num_points, embed_dim)
        
        # 聚合为整体特征（平均池化）
        line_features = point_features.mean(dim=1)  # (N, embed_dim)
        
        return line_features
    
    def forward(self, pred_points, gt_points):
        """
        可学习的点匹配
        
        Args:
            pred_points: (N, num_points, 2)
            gt_points: (M, num_points, 2)
        
        Returns:
            matched_points: (N, num_points, 2)
            confidence: (N, 1)
            matching_matrix: (N, M) 匹配概率
        """
        N = pred_points.shape[0]
        M = gt_points.shape[0]
        
        if M == 0:
            return pred_points, torch.zeros(N, 1, device=pred_points.device), None
        
        # 编码点集
        pred_features = self.encode_points(pred_points)  # (N, embed_dim)
        gt_features = self.encode_points(gt_points)      # (M, embed_dim)
        
        # 计算匹配得分矩阵
        matching_scores = torch.zeros(N, M, device=pred_points.device)
        
        for i in range(N):
            for j in range(M):
                # 拼接特征对
                pair_feat = torch.cat([pred_features[i], gt_features[j]])
                # 计算匹配得分
                matching_scores[i, j] = self.matching_net(pair_feat).squeeze()
        
        # Softmax归一化
        matching_probs = F.softmax(matching_scores, dim=1)  # (N, M)
        
        # 软匹配：加权平均
        matched_points = torch.einsum('nm,mpd->npd', matching_probs, gt_points)
        
        # 计算置信度
        max_matching_prob = matching_probs.max(dim=1)[0]  # (N,)
        confidence = max_matching_prob.unsqueeze(-1)  # (N, 1)
        
        return matched_points, confidence, matching_probs


class HungarianPointMatcher(nn.Module):
    """
    使用Hungarian算法的点匹配器（全局最优）
    
    Args:
        num_points (int): 每条线的点数
    """
    
    def __init__(self, num_points=20):
        super().__init__()
        self.num_points = num_points
    
    def forward(self, pred_points, gt_points):
        """
        Hungarian匹配
        
        Args:
            pred_points: (N, num_points, 2)
            gt_points: (M, num_points, 2)
        
        Returns:
            matched_points: (N, num_points, 2)
            confidence: (N, 1)
        """
        from scipy.optimize import linear_sum_assignment
        
        N = pred_points.shape[0]
        M = gt_points.shape[0]
        
        if M == 0 or N == 0:
            return pred_points, torch.zeros(N, 1, device=pred_points.device)
        
        # 计算代价矩阵
        cost_matrix = torch.zeros(N, M)
        for i in range(N):
            for j in range(M):
                cost = (pred_points[i] - gt_points[j]).pow(2).sum(-1).sqrt().mean()
                cost_matrix[i, j] = cost
        
        # Hungarian匹配
        if N <= M:
            row_ind, col_ind = linear_sum_assignment(cost_matrix.cpu().numpy())
            
            # 构建匹配结果
            matched_points = torch.zeros_like(pred_points)
            confidence = torch.zeros(N, 1, device=pred_points.device)
            
            for i, j in zip(row_ind, col_ind):
                matched_points[i] = gt_points[j]
                confidence[i] = torch.exp(-cost_matrix[i, j] / 1.0)
        else:
            # N > M: 有些pred没有对应的GT
            row_ind, col_ind = linear_sum_assignment(cost_matrix.cpu().numpy())
            
            matched_points = pred_points.clone()
            confidence = torch.zeros(N, 1, device=pred_points.device)
            
            for i, j in zip(row_ind, col_ind):
                matched_points[i] = gt_points[j]
                confidence[i] = torch.exp(-cost_matrix[i, j] / 1.0)
        
        return matched_points, confidence

