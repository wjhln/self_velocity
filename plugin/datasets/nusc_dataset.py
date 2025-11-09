from.base_dataset import BaseMapDataset
from .map_utils.nuscmap_extractor import NuscMapExtractor
from mmdet.datasets import DATASETS
import numpy as np
from .visualize.renderer import Renderer
import mmcv
from time import time
from pyquaternion import Quaternion
import math

@DATASETS.register_module()
class NuscDataset(BaseMapDataset):
    """NuScenes map dataset class.

    Args:
        ann_file (str): annotation file path
        cat2id (dict): category to class id
        roi_size (tuple): bev range
        eval_config (Config): evaluation config
        meta (dict): meta information
        pipeline (Config): data processing pipeline config
        interval (int): annotation load interval
        work_dir (str): path to work dir
        test_mode (bool): whether in test mode
    """
    
    def __init__(self, data_root, **kwargs):
        super().__init__(**kwargs)
        self.map_extractor = NuscMapExtractor(data_root, self.roi_size)
        self.renderer = Renderer(self.cat2id, self.roi_size, 'nusc')
    
    def load_annotations(self, ann_file):
        """Load annotations from ann_file.

        Args:
            ann_file (str): Path of the annotation file.

        Returns:
            list[dict]: List of annotations.
        """
        
        start_time = time()
        ann = mmcv.load(ann_file)
        samples = ann[::self.interval]
        
        print(f'collected {len(samples)} samples in {(time() - start_time):.2f}s')
        self.samples = samples

    def get_sample(self, idx):
        """Get data sample. For each sample, map extractor will be applied to extract 
        map elements. 
        
        Args:
            idx (int): data index

        Returns:
            result (dict): dict of input
        """

        sample = self.samples[idx]
        location = sample['location']
        
        map_geoms = self.map_extractor.get_map_geom(location, sample['e2g_translation'], 
                sample['e2g_rotation'])

        map_label2geom = {}
        for k, v in map_geoms.items():
            if k in self.cat2id.keys():
                map_label2geom[self.cat2id[k]] = v
        
        ego2img_rts = []
        for c in sample['cams'].values():
            extrinsic, intrinsic = np.array(
                c['extrinsics']), np.array(c['intrinsics'])
            ego2cam_rt = extrinsic
            viewpad = np.eye(4)
            viewpad[:intrinsic.shape[0], :intrinsic.shape[1]] = intrinsic
            ego2cam_rt = (viewpad @ ego2cam_rt)
            ego2img_rts.append(ego2cam_rt)

        # 计算速度信息（从位姿变化）
        velocity_info = self._compute_velocity(idx)

        # if sample['sample_idx'] == 0:
        #     is_first_frame = True
        # else:
        #     is_first_frame = self.flag[sample['sample_idx']] > self.flag[sample['sample_idx'] - 1]
        input_dict = {
            'location': location,
            'token': sample['token'],
            'img_filenames': [c['img_fpath'] for c in sample['cams'].values()],
            # intrinsics are 3x3 Ks
            'cam_intrinsics': [c['intrinsics'] for c in sample['cams'].values()],
            # extrinsics are 4x4 tranform matrix, **ego2cam**
            'cam_extrinsics': [c['extrinsics'] for c in sample['cams'].values()],
            'ego2img': ego2img_rts,
            'map_geoms': map_label2geom, # {0: List[ped_crossing(LineString)], 1: ...}
            'ego2global_translation': sample['e2g_translation'], 
            'ego2global_rotation': Quaternion(sample['e2g_rotation']).rotation_matrix.tolist(),
            # 'is_first_frame': is_first_frame, # deprecated
            'sample_idx': sample['sample_idx'],
            'scene_name': sample['scene_name'],
            # 新增: 速度信息
            'velocity': velocity_info['velocity'],  # [vx, vy, vz] in ego frame
            'velocity_magnitude': velocity_info['magnitude'],  # |v|
            'timestamp': sample['timestamp']
            # 'group_idx': self.flag[sample['sample_idx']]
        }

        return input_dict
    
    def _compute_velocity(self, idx):
        """
        从位姿变化计算速度
        
        Args:
            idx: 当前样本索引
        
        Returns:
            dict: 包含速度信息
                - velocity: [vx, vy, vz] 在ego坐标系下
                - magnitude: 速度大小
        """
        sample_curr = self.samples[idx]
        
        # 查找下一帧（同场景）
        next_idx = idx + 1
        if next_idx >= len(self.samples):
            # 最后一帧，返回零速度
            return {
                'velocity': [0.0, 0.0, 0.0],
                'magnitude': 0.0
            }
        
        sample_next = self.samples[next_idx]
        
        # 检查是否同一场景
        if sample_curr.get('scene_name') != sample_next.get('scene_name'):
            return {
                'velocity': [0.0, 0.0, 0.0],
                'magnitude': 0.0
            }
        
        # 计算时间差
        dt = (sample_next['timestamp'] - sample_curr['timestamp']) / 1e6  # 微秒转秒
        
        if dt < 0.01 or dt > 2.0:  # 时间间隔不合理
            return {
                'velocity': [0.0, 0.0, 0.0],
                'magnitude': 0.0
            }
        
        # 全局坐标系下的位移
        pos_curr = np.array(sample_curr['e2g_translation'])
        pos_next = np.array(sample_next['e2g_translation'])
        global_displacement = pos_next - pos_curr
        
        # 全局坐标系下的速度
        global_velocity = global_displacement / dt
        
        # 转换到当前帧的ego坐标系
        rot_curr = Quaternion(sample_curr['e2g_rotation']).rotation_matrix
        ego_velocity = rot_curr.T @ global_velocity
        
        velocity_magnitude = np.linalg.norm(ego_velocity[:2])  # xy平面速度
        
        return {
            'velocity': ego_velocity.tolist(),
            'magnitude': float(velocity_magnitude)
        }