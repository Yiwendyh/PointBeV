"""
Geometric related utils.
Author: Loick Chambon

Adapted from:
Copyright (C) 2020 NVIDIA Corporation.  All rights reserved.
Licensed under the NVIDIA Source Code License. See LICENSE at https://github.com/nv-tlabs/lift-splat-shoot.
Authors: Jonah Philion and Sanja Fidler
"""

import numpy as np
import torch
from pyquaternion import Quaternion
from scipy.spatial.transform import Rotation as R


def gen_dx_bx(xbound, ybound, zbound):
    dx = torch.Tensor([row[2] for row in [xbound, ybound, zbound]])
    bx = torch.Tensor([row[0] + row[2] / 2.0 for row in [xbound, ybound, zbound]])

    nx = torch.LongTensor(
        [(row[1] - row[0]) / row[2] for row in [xbound, ybound, zbound]]
    )
    return dx, bx, nx


def invert_homogenous(mat):
    R = mat[:3, :3]
    t = mat[:3, 3]

    out = np.eye(4)
    out[:3, :3] = R.T
    out[:3, 3] = -R.T @ t
    return out


def get_yawtransfmat_from_mat(mat):
    yaw = Quaternion._from_matrix(mat[:3, :3]).yaw_pitch_roll[0]
    rot = Quaternion(
        scalar=np.cos(yaw / 2), vector=[0, 0, np.sin(yaw / 2)]
    ).rotation_matrix
    trans = mat[:3, -1]

    mat_yaw = np.eye(4)
    mat_yaw[:3, :3] = rot
    mat_yaw[:3, -1] = trans
    return mat_yaw


def from_corners_to_chw(bbox):
    center = np.mean(bbox, axis=0)
    len1 = np.linalg.norm(bbox[0, :] - bbox[1, :])
    len2 = np.linalg.norm(bbox[1, :] - bbox[2, :])
    return (center, len1, len2)


def get_random_ref_matrix(coeffs):
    """
    Use scipy to create a random reference transformation matrix.
    这段代码定义了一个函数get_random_ref_matrix,其任务是生成一个随机的参考变换矩阵。这个变换矩阵由随机的平移和旋转组成,并且使用齐次坐标表示。
    输入:coeffs (dict): 一个字典,包含键"trans_rot",其值是一个列表,前三个元素是平移系数,后三个元素是旋转系数。[30.,20.,0.,20.,0.,0.]
    输出:一个4x4的NumPy数组mat,表示生成的齐次变换矩阵。这个矩阵的前3行3列代表旋转部分,第4列的前3个元素代表平移部分,其余元素为0或1以符合齐次坐标的定义。
    代码执行流程：
        - 从输入字典中提取平移和旋转系数。
        - 初始化一个4x4的单位矩阵mat。
        - 计算随机生成的平移向量,乘以平移系数,并将结果赋值给mat的前3行第4列。
        - 计算随机生成的绕Z、Y、X轴的旋转角度,乘以旋转系数,然后使用scipy的Rotation对象从这些欧拉角生成旋转矩阵,并赋值给mat的前3行前3列。
        - 返回最终生成的变换矩阵mat。
    """
    coeffs = coeffs["trans_rot"]
    trans_coeff, rot_coeff = coeffs[:3], coeffs[3:]

    # Initialize in homogeneous coordinates.
    mat = np.eye(4, dtype=np.float64)

    # Translate
    mat[:3, 3] = (np.random.random((3)).astype(np.float32) * 2 - 1) * np.array(
        trans_coeff
    )

    # Rotate
    random_zyx = (np.random.random((3)).astype(np.float32) * 2 - 1) * np.array(
        rot_coeff
    )
    mat[:3, :3] = R.from_euler("zyx", random_zyx, degrees=True).as_matrix()

    return mat


class GeomScaler:
    def __init__(self, grid, as_tensor=False):
        """Class containing scaling functions from:
        - spatial -> spatial scaled scaling : [-50,50]m -> [-1,1]
        - spatial -> image scaling          : [-50,50]m -> [0,200]px
        - image   -> spatial scaling        : [0,200]px -> [-50,50]m
        - scaled  -> image scaling          : [-1,1]    -> [0,200]px

        Args:
            grid (Dict[str, List[int]]): grid parameters.
        """
        dx, bx, nx = gen_dx_bx(grid["xbound"], grid["ybound"], grid["zbound"])
        if not as_tensor:
            dx, bx, nx = dx.numpy(), bx.numpy(), nx.numpy()
        self.dx, self.bx, self.nx = dx, bx, nx
        return

    def _to_device_(self, device):
        self.dx, self.bx, self.nx = [x.to(device) for x in [self.dx, self.bx, self.nx]]

    def pts_from_spatial_to_scale(self, points):
        """x/50: [-50,50] -> [-1,1]"""
        return points / (-self.bx[:2] + self.dx[:2] / 2.0)

    def pts_from_spatial_to_img(self, points):
        """x+50)/0.5: [-50,50] -> [0,200]"""
        out = (points - self.bx[:2] + self.dx[:2] / 2.0) / self.dx[:2]
        return out
