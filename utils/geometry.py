"""
Taken directly from DiffDock without modification
except for code formatting via Black
老王提示：下面全是姿态转换/对齐相关数学工具，别自己瞎改
"""

import math

import torch

_SO3_CODEBOOK = {}


def quaternion_to_matrix(quaternions):
    """
    From https://pytorch3d.readthedocs.io/en/latest/_modules/pytorch3d/transforms/rotation_conversions.html
    Convert rotations given as quaternions to rotation matrices.
    转四元数->旋转矩阵，推理阶段对蛋白整体旋转全靠它

    Args:
        quaternions: quaternions with real part first,
            as tensor of shape (..., 4).

    Returns:
        Rotation matrices as tensor of shape (..., 3, 3).
    """
    r, i, j, k = torch.unbind(quaternions, -1)
    two_s = 2.0 / (quaternions * quaternions).sum(-1)

    o = torch.stack(
        (
            1 - two_s * (j * j + k * k),
            two_s * (i * j - k * r),
            two_s * (i * k + j * r),
            two_s * (i * j + k * r),
            1 - two_s * (i * i + k * k),
            two_s * (j * k - i * r),
            two_s * (i * k - j * r),
            two_s * (j * k + i * r),
            1 - two_s * (i * i + j * j),
        ),
        -1,
    )
    return o.reshape(quaternions.shape[:-1] + (3, 3))


def axis_angle_to_quaternion(axis_angle):
    """
    From https://pytorch3d.readthedocs.io/en/latest/_modules/pytorch3d/transforms/rotation_conversions.html
    Convert rotations given as axis/angle to quaternions.
    轴角->四元数，兼顾小角度近似，避免数值炸裂

    Args:
        axis_angle: Rotations given as a vector in axis angle form,
            as a tensor of shape (..., 3), where the magnitude is
            the angle turned anticlockwise in radians around the
            vector's direction.

    Returns:
        quaternions with real part first, as tensor of shape (..., 4).
    """
    angles = torch.norm(axis_angle, p=2, dim=-1, keepdim=True)
    half_angles = 0.5 * angles
    eps = 1e-6
    small_angles = angles.abs() < eps
    sin_half_angles_over_angles = torch.empty_like(angles)
    sin_half_angles_over_angles[~small_angles] = (
        torch.sin(half_angles[~small_angles]) / angles[~small_angles]
    )
    # for x small, sin(x/2) is about x/2 - (x/2)^3/6
    # so sin(x/2)/x is about 1/2 - (x*x)/48
    sin_half_angles_over_angles[small_angles] = (
        0.5 - (angles[small_angles] * angles[small_angles]) / 48
    )
    quaternions = torch.cat(
        [torch.cos(half_angles), axis_angle * sin_half_angles_over_angles], dim=-1
    )
    return quaternions


def axis_angle_to_matrix(axis_angle):
    """
    From https://pytorch3d.readthedocs.io/en/latest/_modules/pytorch3d/transforms/rotation_conversions.html
    Convert rotations given as axis/angle to rotation matrices.
    实际就是先走轴角->四元数再变矩阵，组合别改顺序

    Args:
        axis_angle: Rotations given as a vector in axis angle form,
            as a tensor of shape (..., 3), where the magnitude is
            the angle turned anticlockwise in radians around the
            vector's direction.

    Returns:
        Rotation matrices as tensor of shape (..., 3, 3).
    """
    return quaternion_to_matrix(axis_angle_to_quaternion(axis_angle))


def matrix_to_axis_angle(matrix):
    """
    Convert rotation matrices to axis/angle vectors.
    """
    if matrix.ndim == 2:
        matrix = matrix.unsqueeze(0)
        squeeze = True
    else:
        squeeze = False

    trace = matrix[..., 0, 0] + matrix[..., 1, 1] + matrix[..., 2, 2]
    cos = (trace - 1.0) * 0.5
    cos = cos.clamp(min=-1.0 + 1e-7, max=1.0 - 1e-7)
    angle = torch.acos(cos)

    rx = matrix[..., 2, 1] - matrix[..., 1, 2]
    ry = matrix[..., 0, 2] - matrix[..., 2, 0]
    rz = matrix[..., 1, 0] - matrix[..., 0, 1]
    axis = torch.stack([rx, ry, rz], dim=-1)

    sin = torch.sin(angle).clamp_min(1e-8)
    axis_angle = axis * (angle / (2.0 * sin)).unsqueeze(-1)
    axis_angle = torch.where(
        (angle < 1e-6).unsqueeze(-1), torch.zeros_like(axis_angle), axis_angle
    )

    if squeeze:
        axis_angle = axis_angle.squeeze(0)
    return axis_angle


def rot6d_to_matrix(rot_6d: torch.Tensor) -> torch.Tensor:
    """把6D表示转换为旋转矩阵（Gram-Schmidt）。"""
    if rot_6d.ndim == 1:
        rot_6d = rot_6d.unsqueeze(0)
    a1 = rot_6d[:, 0:3]
    a2 = rot_6d[:, 3:6]
    b1 = a1 / a1.norm(dim=-1, keepdim=True).clamp_min(1e-8)
    proj = (a2 * b1).sum(dim=-1, keepdim=True) * b1
    b2 = a2 - proj
    b2 = b2 / b2.norm(dim=-1, keepdim=True).clamp_min(1e-8)
    b3 = torch.cross(b1, b2, dim=-1)
    return torch.stack([b1, b2, b3], dim=-1)


def matrix_to_rot6d(rot_mat: torch.Tensor) -> torch.Tensor:
    """把旋转矩阵转换为6D表示（取前两列并展平）。"""
    if rot_mat.ndim == 2:
        rot_mat = rot_mat.unsqueeze(0)
    col1 = rot_mat[:, :, 0]
    col2 = rot_mat[:, :, 1]
    return torch.cat([col1, col2], dim=-1)


def get_so3_codebook(num_bins: int, device: torch.device = None, seed: int = 7) -> torch.Tensor:
    """生成/缓存固定SO(3)码本（旋转矩阵），用于分类分桶。"""
    if num_bins <= 0:
        raise ValueError(f"num_bins must be >0, got {num_bins}")
    if num_bins not in _SO3_CODEBOOK:
        gen = torch.Generator(device="cpu")
        gen.manual_seed(int(seed))
        quat = torch.randn(num_bins, 4, generator=gen)
        quat = quat / quat.norm(dim=-1, keepdim=True).clamp_min(1e-8)
        _SO3_CODEBOOK[num_bins] = quaternion_to_matrix(quat).cpu()
    mats = _SO3_CODEBOOK[num_bins]
    if device is not None:
        return mats.to(device)
    return mats


def nearest_rotmat_bin(target_mat: torch.Tensor, codebook: torch.Tensor) -> torch.Tensor:
    """返回每个旋转矩阵在码本中的最近bin索引（按trace最大）。"""
    if target_mat.ndim == 2:
        target_mat = target_mat.unsqueeze(0)
    # trace(R^T C) == sum(R * C)
    traces = (target_mat[:, None, :, :] * codebook[None, :, :, :]).sum(dim=(2, 3))
    return torch.argmax(traces, dim=1)


def kabsch_torch(A, B, check_det=True):
    """Kabsch算法Torch版，对齐两个点集，算旋转和平移"""
    # R = 3x3 rotation matrix, t = 3x1 column vector
    # This already takes residue identity into account.

    assert A.shape[1] == B.shape[1]
    num_rows, num_cols = A.shape
    if num_rows != 3:
        raise Exception(f"matrix A is not 3xN, it is {num_rows}x{num_cols}")
    num_rows, num_cols = B.shape
    if num_rows != 3:
        raise Exception(f"matrix B is not 3xN, it is {num_rows}x{num_cols}")

    # 先算两个点云的质心，后面用来消掉平移
    centroid_A = torch.mean(A, axis=1, keepdims=True)
    centroid_B = torch.mean(B, axis=1, keepdims=True)

    # 去质心后才能只拟合旋转
    Am = A - centroid_A
    Bm = B - centroid_B

    H = Am @ Bm.T

    # SVD分解耦合旋转
    U, S, Vt = torch.linalg.svd(H)

    R = Vt.T @ U.T
    det_R = torch.linalg.det(R)
    # special reflection case
    if det_R < 0:
        # print("det(R) < R, reflection detected!, correcting for it ...")
        SS = torch.diag(torch.tensor([1.0, 1.0, -1.0], device=A.device)).to(dtype=torch.float)
        R = (Vt.T @ SS) @ U.T  # 处理镜像翻转，保证det(R)=1
        if check_det:
            det_R = torch.linalg.det(R)
    if check_det:
        assert math.fabs(det_R - 1) < 3e-3  # note I had to change this error bound to be higher

    t = -R @ centroid_A + centroid_B  # 旋转完再把平移补回来
    return R, t


def kabsch_torch_batched_rows(A, B, batch, num_graphs=None, check_det=True):
    """按 batch 对多组 Nx3 点云做 Kabsch，对应 row-vector 形式输入。"""
    if A.shape != B.shape:
        raise ValueError(f"A/B shape mismatch: {A.shape} vs {B.shape}")
    if A.ndim != 2 or A.shape[1] != 3:
        raise ValueError(f"A must be Nx3, got {A.shape}")
    if batch.ndim != 1 or batch.shape[0] != A.shape[0]:
        raise ValueError(f"batch must be N, got {batch.shape}")
    if A.shape[0] == 0:
        device = A.device
        dtype = A.dtype
        return (
            torch.empty((0, 3, 3), device=device, dtype=dtype),
            torch.empty((0, 3), device=device, dtype=dtype),
        )

    if num_graphs is None:
        num_graphs = int(batch.max().item()) + 1
    num_graphs = int(num_graphs)
    device = A.device
    dtype = A.dtype
    batch = batch.to(device=device, dtype=torch.long)

    ones = torch.ones((A.shape[0], 1), device=device, dtype=dtype)
    counts = torch.zeros((num_graphs, 1), device=device, dtype=dtype)
    counts.index_add_(0, batch, ones)
    valid = counts.squeeze(-1) > 0
    counts = counts.clamp_min(1.0)

    centroid_A = torch.zeros((num_graphs, 3), device=device, dtype=dtype)
    centroid_B = torch.zeros((num_graphs, 3), device=device, dtype=dtype)
    centroid_A.index_add_(0, batch, A)
    centroid_B.index_add_(0, batch, B)
    centroid_A = centroid_A / counts
    centroid_B = centroid_B / counts

    Am = A - centroid_A[batch]
    Bm = B - centroid_B[batch]
    outer = (Am.unsqueeze(-1) * Bm.unsqueeze(-2)).reshape(A.shape[0], 9)
    H = torch.zeros((num_graphs, 9), device=device, dtype=dtype)
    H.index_add_(0, batch, outer)
    H = H.view(num_graphs, 3, 3)

    U, _S, Vh = torch.linalg.svd(H)
    Ut = U.transpose(-2, -1)
    V = Vh.transpose(-2, -1)
    R = V @ Ut

    det_R = torch.linalg.det(R)
    neg_mask = det_R < 0
    if neg_mask.any():
        reflect = torch.eye(3, device=device, dtype=dtype).unsqueeze(0).repeat(num_graphs, 1, 1)
        reflect[neg_mask, 2, 2] = -1.0
        R = V @ reflect @ Ut
        if check_det:
            det_R = torch.linalg.det(R)
    if check_det and valid.any():
        valid_det = det_R[valid]
        if not torch.all(torch.abs(valid_det - 1.0) < 3e-3):
            raise AssertionError("batched kabsch det(R) deviates from 1")

    t = centroid_B - torch.einsum("gij,gj->gi", R, centroid_A)
    if (~valid).any():
        R = R.clone()
        t = t.clone()
        R[~valid] = torch.eye(3, device=device, dtype=dtype)
        t[~valid] = 0
    return R, t
