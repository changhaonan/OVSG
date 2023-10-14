""" Spatial relationship judger. """
from __future__ import annotations
from enum import Enum
import random
import numpy as np
import open3d as o3d
import tqdm
from clip import clip

# Global variables
NUM_BASE_REL = 9


# Utils methods
def rotate_point(point: np.ndarray, theta: float):
    """rotate the point by theta along z-axis."""
    theta = -theta  # rotate counter-clockwise
    x, y = point
    # rotate
    xy = np.array([x, y])
    rotate_matrix = np.array([[np.cos(theta), -np.sin(theta)], [np.sin(theta), np.cos(theta)]])
    xy = rotate_matrix @ xy
    # return
    return np.array([xy[0], xy[1]])


def rotate_pose(pose: np.ndarray, theta: float):
    """rotate the pose by theta along z-axis."""
    xc, yc, zc, xl, yf, zl, xr, yb, zu = pose
    # rotate
    xyf0 = np.array([xl, yf])
    xyf2 = np.array([xl, yb])
    xyb2 = np.array([xr, yb])
    xyb0 = np.array([xr, yf])
    xyc1 = np.array([xc, yc])
    # rotate
    xyf0 = rotate_point(xyf0, theta)
    xyf2 = rotate_point(xyf2, theta)
    xyb2 = rotate_point(xyb2, theta)
    xyb0 = rotate_point(xyb0, theta)
    xyc1 = rotate_point(xyc1, theta)
    # new pose
    xl = min(xyf0[0], xyf2[0], xyb2[0], xyb0[0])
    yf = min(xyf0[1], xyf2[1], xyb2[1], xyb0[1])
    zl = zl
    xc = xyc1[0]
    yc = xyc1[1]
    zc = zc
    xr = max(xyf0[0], xyf2[0], xyb2[0], xyb0[0])
    yb = max(xyf0[1], xyf2[1], xyb2[1], xyb0[1])
    zu = zu
    # return
    pose_new = np.array([xc, yc, zc, xl, yf, zl, xr, yb, zu])
    return pose_new


def iou_xy(pose_a: np.ndarray, pose_b: np.ndarray):
    """Compute the iou of two poses for xy plane"""
    xc_a, yc_a, zc_a, xl_a, yf_a, zl_a, xr_a, yb_a, zu_a = pose_a
    xc_b, yc_b, zc_b, xl_b, yf_b, zl_b, xr_b, yb_b, zu_b = pose_b
    # compute iou
    x1 = max(xl_a, xl_b)
    y1 = max(yf_a, yf_b)
    x2 = min(xr_a, xr_b)
    y2 = min(yb_a, yb_b)
    w = max(0, x2 - x1)
    h = max(0, y2 - y1)
    inter = w * h
    area_a = (xr_a - xl_a) * (yb_a - yf_a)
    area_b = (xr_b - xl_b) * (yb_b - yf_b)
    iou_a = inter / area_a
    iou_b = inter / area_b
    return iou_a, iou_b


def iou_xyz(pose_a: np.ndarray, pose_b: np.ndarray):
    """Compute the iou of two poses for xyz space"""
    xc_a, yc_a, zc_a, xl_a, yf_a, zl_a, xr_a, yb_a, zu_a = pose_a
    xc_b, yc_b, zc_b, xl_b, yf_b, zl_b, xr_b, yb_b, zu_b = pose_b
    # compute iou
    x1 = max(xl_a, xl_b)
    y1 = max(yf_a, yf_b)
    z1 = max(zl_a, zl_b)
    x2 = min(xr_a, xr_b)
    y2 = min(yb_a, yb_b)
    z2 = min(zu_a, zu_b)
    w = max(0, x2 - x1)
    h = max(0, y2 - y1)
    d = max(0, z2 - z1)
    inter = w * h * d
    area_a = (xr_a - xl_a) * (yb_a - yf_a) * (zu_a - zl_a)
    area_b = (xr_b - xl_b) * (yb_b - yf_b) * (zu_b - zl_b)
    iou_a = inter / area_a
    iou_b = inter / area_b
    return iou_a, iou_b


# Constants
OVERLAP_THRESH = 0.8


class Points9:
    """Represent pose with 9 points."""

    # Interface methods
    @staticmethod
    def from_pcd(pcd: o3d.geometry.PointCloud | np.ndarray):
        """Parse the point cloud into 9 points."""
        if isinstance(pcd, o3d.geometry.PointCloud):
            points = np.asarray(pcd.points)
        else:
            points = pcd
        xc, yc, zc = np.mean(points, axis=0)
        xl, yf, zl = np.min(points, axis=0)
        xr, yb, zu = np.max(points, axis=0)
        return np.array([xc, yc, zc, xl, yf, zl, xr, yb, zu])

    @staticmethod
    def from_bbox(pos: np.ndarray, size: np.ndarray, center: np.ndarray | None = None):
        """Parse the bounding box into 9 points."""
        if center is None:
            center = pos
        xc, yc, zc = center
        xl, yf, zl = pos - size / 2
        xr, yb, zu = pos + size / 2
        return np.array([xc, yc, zc, xl, yf, zl, xr, yb, zu])

    @staticmethod
    def to_geometry(
        pos_p9: np.ndarray, color=None
    ) -> tuple[o3d.geometry.AxisAlignedBoundingBox, o3d.geometry.TriangleMesh]:
        """Parse the 9 points into bounding box & sphere."""
        if color is None:
            # random color
            color = np.random.rand(3)
            color = color * 0.5 + 0.5  # make it brighter
        xc, yc, zc, xl, yf, zl, xr, yb, zu = pos_p9
        # bounding box
        pos = np.array([xc, yc, zc])
        min_bound = np.array([xl, yf, zl])
        max_bound = np.array([xr, yb, zu])
        bbox = o3d.geometry.AxisAlignedBoundingBox(min_bound=min_bound, max_bound=max_bound)
        bbox.color = color
        # sphere
        radius = 0.05 * max(xr - xl, yb - yf, zu - zl)
        sphere = o3d.geometry.TriangleMesh.create_sphere(radius=radius)
        sphere.translate(pos)
        sphere.paint_uniform_color(color)
        # return
        return bbox, sphere

    @staticmethod
    def zeros():
        """Zero pose by 9 points, (xl, xc, xr, yf, yc, yb, zl, zc, zu)."""
        return np.zeros(9)

    @staticmethod
    def random_poses(num_sample: int):
        """Random pose by 9 points, (xl, xc, xr, yf, yc, yb, zl, zc, zu)."""
        points_c = np.random.rand(num_sample, 3)  # (xc, yc, zc)
        points_l = points_c - np.random.rand(num_sample, 3)  # (xl, yf, zl)
        points_r = points_c + np.random.rand(num_sample, 3)  # (xr, yb, zu)
        poses = np.concatenate((points_c, points_l, points_r), axis=1)
        return poses

    @staticmethod
    def random_pose_pair(num_sample: int, fliter_none: bool = True):
        """Random pose pair by 9 points, (xl, xc, xr, yf, yc, yb, zl, zc, zu)."""
        poses_a = Points9.random_poses(num_sample)
        poses_b = Points9.random_poses(num_sample)
        # filter pairs that has no relationship
        pair_valid = np.ones(num_sample, dtype=bool)
        if fliter_none:
            for i, (pose_a, pose_b) in enumerate(zip(poses_a, poses_b)):
                label = Points9.label(pose_a, pose_b)
                if label.sum() == 0:
                    pair_valid[i] = False
        poses_a = poses_a[pair_valid]
        poses_b = poses_b[pair_valid]
        return poses_a, poses_b

    @staticmethod
    def normalize_pair(pose_a: np.ndarray, pose_b: np.ndarray):
        """Normalize the pose of object A and B into a unit cube."""
        # normalize
        xc_a, yc_a, zc_a, xl_a, yf_a, zl_a, xr_a, yb_a, zu_a = pose_a
        xc_b, yc_b, zc_b, xl_b, yf_b, zl_b, xr_b, yb_b, zu_b = pose_b
        # normalize x
        x_max = max(xr_a, xr_b)
        x_min = min(xl_a, xl_b)
        scale = x_max - x_min
        offset = x_min
        xl_a = (xl_a - offset) / scale
        xc_a = (xc_a - offset) / scale
        xr_a = (xr_a - offset) / scale
        xl_b = (xl_b - offset) / scale
        xc_b = (xc_b - offset) / scale
        xr_b = (xr_b - offset) / scale
        # normalize y
        y_max = max(yb_a, yb_b)
        y_min = min(yf_a, yf_b)
        scale = y_max - y_min
        offset = y_min
        yf_a = (yf_a - offset) / scale
        yc_a = (yc_a - offset) / scale
        yb_a = (yb_a - offset) / scale
        yf_b = (yf_b - offset) / scale
        yc_b = (yc_b - offset) / scale
        yb_b = (yb_b - offset) / scale
        # normalize z
        z_max = max(zu_a, zu_b)
        z_min = min(zl_a, zl_b)
        scale = z_max - z_min
        offset = z_min
        zl_a = (zl_a - offset) / scale
        zc_a = (zc_a - offset) / scale
        zu_a = (zu_a - offset) / scale
        zl_b = (zl_b - offset) / scale
        zc_b = (zc_b - offset) / scale
        zu_b = (zu_b - offset) / scale
        # return
        return np.array(
            [
                xc_a,
                yc_a,
                zc_a,
                xl_a,
                yf_a,
                zl_a,
                xr_a,
                yb_a,
                zu_a,
                xc_b,
                yc_b,
                zc_b,
                xl_b,
                yf_b,
                zl_b,
                xr_b,
                yb_b,
                zu_b,
            ]
        )

    @staticmethod
    def normalize_pairs(pose_a: np.ndarray, pose_b: np.ndarray):
        """Normalize the poses of objects A and B into a unit cube."""

        assert (
            pose_a.shape[1] == 9 and pose_b.shape[1] == 9
        ), "Both input arrays should have a shape of (N, 9)"

        # Extract coordinates
        xc_a, yc_a, zc_a, xl_a, yf_a, zl_a, xr_a, yb_a, zu_a = np.split(pose_a, 9, axis=1)
        xc_b, yc_b, zc_b, xl_b, yf_b, zl_b, xr_b, yb_b, zu_b = np.split(pose_b, 9, axis=1)

        # Normalize x
        x_max = np.maximum(xr_a, xr_b)
        x_min = np.minimum(xl_a, xl_b)
        scale_x = x_max - x_min
        offset_x = x_min
        xl_a = (xl_a - offset_x) / scale_x
        xc_a = (xc_a - offset_x) / scale_x
        xr_a = (xr_a - offset_x) / scale_x
        xl_b = (xl_b - offset_x) / scale_x
        xc_b = (xc_b - offset_x) / scale_x
        xr_b = (xr_b - offset_x) / scale_x

        # Normalize y
        y_max = np.maximum(yb_a, yb_b)
        y_min = np.minimum(yf_a, yf_b)
        scale_y = y_max - y_min
        offset_y = y_min
        yf_a = (yf_a - offset_y) / scale_y
        yc_a = (yc_a - offset_y) / scale_y
        yb_a = (yb_a - offset_y) / scale_y
        yf_b = (yf_b - offset_y) / scale_y
        yc_b = (yc_b - offset_y) / scale_y
        yb_b = (yb_b - offset_y) / scale_y

        # Normalize z
        z_max = np.maximum(zu_a, zu_b)
        z_min = np.minimum(zl_a, zl_b)
        scale_z = z_max - z_min
        offset_z = z_min
        zl_a = (zl_a - offset_z) / scale_z
        zc_a = (zc_a - offset_z) / scale_z
        zu_a = (zu_a - offset_z) / scale_z
        zl_b = (zl_b - offset_z) / scale_z
        zc_b = (zc_b - offset_z) / scale_z
        zu_b = (zu_b - offset_z) / scale_z

        # Return
        return np.concatenate(
            [
                xc_a,
                yc_a,
                zc_a,
                xl_a,
                yf_a,
                zl_a,
                xr_a,
                yb_a,
                zu_a,
                xc_b,
                yc_b,
                zc_b,
                xl_b,
                yf_b,
                zl_b,
                xr_b,
                yb_b,
                zu_b,
            ],
            axis=1,
        )

    # Debug methods
    @staticmethod
    def visualize(poses: np.ndarray):
        """Visualize the random relation."""
        mesh_list = []
        # create origin axis
        origin = o3d.geometry.TriangleMesh.create_coordinate_frame(size=1.0)
        mesh_list.append(origin)

        # object A
        pose_a = poses[0:9]
        bbox, sphere = Points9.to_geometry(pose_a)
        sphere.paint_uniform_color(np.array([1, 0, 0]))
        bbox.color = np.array([1, 0, 0])  # red
        mesh_list.append(sphere)
        mesh_list.append(bbox)

        # object B
        pose_b = poses[9:18]
        bbox, sphere = Points9.to_geometry(pose_b)
        sphere.paint_uniform_color(np.array([0, 1, 0]))
        bbox.color = np.array([0, 1, 0])  # green
        mesh_list.append(sphere)
        mesh_list.append(bbox)

        # # create bbox of (-1, 1) by (-1, 1) by (-1, 1)
        # bbox = o3d.geometry.AxisAlignedBoundingBox(min_bound=[0, 0, 0], max_bound=[1, 1, 1])
        # # color
        # bbox.color = [0, 0, 1]
        # mesh_list.append(bbox)

        o3d.visualization.draw_geometries(mesh_list)

    # Basic directions
    @staticmethod
    def is_left(pose_a: np.ndarray, pose_b: np.ndarray, theta: float = 0.0):
        """Judge whether object A is left of object B with view angle rotated by theta."""
        xc_a, yc_a, zc_a, xl_a, yf_a, zl_a, xr_a, yb_a, zu_a = rotate_pose(pose_a, theta)
        xc_b, yc_b, zc_b, xl_b, yf_b, zl_b, xr_b, yb_b, zu_b = rotate_pose(pose_b, theta)
        # compute x_13_b
        x_13_b = xl_b + (xr_b - xl_b) / 3.0
        # compute iou_left
        union_left = min(xr_a, x_13_b) - min(xl_a, x_13_b)
        if union_left > (xr_a - xl_a) / 2.0 and xr_a < (xr_b + xl_b) / 2.0:
            return True
        else:
            return False

    @staticmethod
    def is_right(pose_a: np.ndarray, pose_b: np.ndarray, theta: float = 0.0):
        """Judge whether object A is right of object B with view angle rotated by theta."""
        xc_a, yc_a, zc_a, xl_a, yf_a, zl_a, xr_a, yb_a, zu_a = rotate_pose(pose_a, theta)
        xc_b, yc_b, zc_b, xl_b, yf_b, zl_b, xr_b, yb_b, zu_b = rotate_pose(pose_b, theta)
        # compute x_23_b
        x_23_b = xl_b + (xr_b - xl_b) * 2.0 / 3.0
        # compute iou_right
        union_right = max(xr_a, x_23_b) - max(xl_a, x_23_b)
        if union_right > (xr_a - xl_a) / 2.0 and xl_a > (xr_b + xl_b) / 2.0:
            return True
        else:
            return False

    @staticmethod
    def is_front(pose_a: np.ndarray, pose_b: np.ndarray, theta: float = 0.0):
        """Judge whether object A is front of object B with view angle rotated by theta."""
        xc_a, yc_a, zc_a, xl_a, yf_a, zl_a, xr_a, yb_a, zu_a = rotate_pose(pose_a, theta)
        xc_b, yc_b, zc_b, xl_b, yf_b, zl_b, xr_b, yb_b, zu_b = rotate_pose(pose_b, theta)
        # compute y_13_b
        y_13_b = yf_b + (yb_b - yf_b) / 3.0
        # compute iou_front
        union_front = min(yb_a, y_13_b) - min(yf_a, y_13_b)
        if union_front > (yb_a - yf_a) / 2.0 and yb_a < (yb_b + yf_b) / 2.0:
            return True
        else:
            return False

    @staticmethod
    def is_behind(pose_a: np.ndarray, pose_b: np.ndarray, theta: float = 0.0):
        """Judge whether object A is behind of object B with view angle rotated by theta."""
        xc_a, yc_a, zc_a, xl_a, yf_a, zl_a, xr_a, yb_a, zu_a = rotate_pose(pose_a, theta)
        xc_b, yc_b, zc_b, xl_b, yf_b, zl_b, xr_b, yb_b, zu_b = rotate_pose(pose_b, theta)
        # compute y_23_b
        y_23_b = yf_b + (yb_b - yf_b) * 2.0 / 3.0
        # compute iou_behind
        union_behind = max(yb_a, y_23_b) - max(yf_a, y_23_b)
        if union_behind > (yb_a - yf_a) / 2.0 and yf_a > (yb_b + yf_b) / 2.0:
            return True
        else:
            return False

    @staticmethod
    def is_on(pose_a: np.ndarray, pose_b: np.ndarray, theta: float = 0.0):
        """Judge whether object A is on object B with view angle rotated by theta."""
        # on means z of A is larger than z of B, and iou between A and B is larger than OVERLAP_THRESH
        xc_a, yc_a, zc_a, xl_a, yf_a, zl_a, xr_a, yb_a, zu_a = pose_a
        xc_b, yc_b, zc_b, xl_b, yf_b, zl_b, xr_b, yb_b, zu_b = pose_b
        # compare iou_a first
        iou_a, iou_b = iou_xy(pose_a, pose_b)
        if iou_a < OVERLAP_THRESH:
            return False
        # compare z-axis, zl_a should be around or larger than zc_b
        zcl_b = (zc_b + zl_b) / 2.0  # middle of zc and zl
        if zl_a < zcl_b:
            return False
        # meanwhile, zl_a should not be too large
        if zl_a > (zu_b + (zu_a - zl_a)):
            return False
        return True

    @staticmethod
    def is_above(pose_a: np.ndarray, pose_b: np.ndarray, theta: float = 0.0):
        """Judge whether object A is on object B with view angle rotated by theta."""
        # on means z of A is larger than z of B, and iou between A and B is larger than OVERLAP_THRESH
        xc_a, yc_a, zc_a, xl_a, yf_a, zl_a, xr_a, yb_a, zu_a = pose_a
        xc_b, yc_b, zc_b, xl_b, yf_b, zl_b, xr_b, yb_b, zu_b = pose_b
        # compare iou_a first
        iou_a, iou_b = iou_xy(pose_a, pose_b)
        if iou_a < OVERLAP_THRESH:
            return False
        # compare z-axis, zl_a should larger than zu_b
        if zl_a < zu_b:
            return False
        return True

    @staticmethod
    def is_in(pose_a: np.ndarray, pose_b: np.ndarray, theta: float = 0.0):
        """Judge whether object A is in object B with view angle rotated by theta."""
        # on means z of A is larger than z of B, and iou between A and B is larger than OVERLAP_THRESH
        xc_a, yc_a, zc_a, xl_a, yf_a, zl_a, xr_a, yb_a, zu_a = pose_a
        xc_b, yc_b, zc_b, xl_b, yf_b, zl_b, xr_b, yb_b, zu_b = pose_b
        # compare iou_a first
        iou_a, iou_b = iou_xy(pose_a, pose_b)
        if iou_a < OVERLAP_THRESH:
            return False
        # compare z-axis, zl_b < zl_a < zu_a < zu_b
        if zl_b > zl_a or zl_a > zu_a or zu_a > zu_b:
            return False
        return True

    @staticmethod
    def is_under(pose_a: np.ndarray, pose_b: np.ndarray, theta: float = 0.0):
        """Judge whether object A is on object B with view angle rotated by theta."""
        # on means z of A is larger than z of B, and iou between A and B is larger than OVERLAP_THRESH
        xc_a, yc_a, zc_a, xl_a, yf_a, zl_a, xr_a, yb_a, zu_a = pose_a
        xc_b, yc_b, zc_b, xl_b, yf_b, zl_b, xr_b, yb_b, zu_b = pose_b
        # compare iou_a first
        iou_a, iou_b = iou_xy(pose_a, pose_b)
        if iou_a < OVERLAP_THRESH:
            return False
        # compare z-axis, zl_a should larger than zu_b
        if zu_a > zl_b:
            return False
        return True

    @staticmethod
    def is_near(pose_a: np.ndarray, pose_b: np.ndarray, theta: float = 0.0):
        """Judge whether object A is near object B with view angle rotated by theta."""
        dist_c = np.linalg.norm(pose_a[:3] - pose_b[:3])
        scale_a = pose_a[6:9] - pose_a[3:6]
        scale_b = pose_b[6:9] - pose_b[3:6]
        scale = np.linalg.norm(scale_a) + np.linalg.norm(scale_b)
        if dist_c < scale:
            return True
        else:
            return False

    @staticmethod
    def iou(pose_a: np.ndarray, pose_b: np.ndarray):
        """Compute iou between two objects."""
        iou_a, iou_b = iou_xyz(pose_a, pose_b)
        return iou_a, iou_b

    # Compose directions
    @staticmethod
    def label(pose_a: np.ndarray, pose_b: np.ndarray, theta: float = 0.0) -> np.ndarray:
        """Label a vector encoding on relation."""
        relation_embeding = []
        relation_embeding.append(Points9.is_left(pose_a, pose_b, theta))
        relation_embeding.append(Points9.is_right(pose_a, pose_b, theta))
        relation_embeding.append(Points9.is_front(pose_a, pose_b, theta))
        relation_embeding.append(Points9.is_behind(pose_a, pose_b, theta))
        relation_embeding.append(Points9.is_on(pose_a, pose_b, theta))
        relation_embeding.append(Points9.is_above(pose_a, pose_b, theta))
        relation_embeding.append(Points9.is_in(pose_a, pose_b, theta))
        relation_embeding.append(Points9.is_under(pose_a, pose_b, theta))
        relation_embeding.append(Points9.is_near(pose_a, pose_b, theta))

        return np.array(relation_embeding).astype(np.int32)

    @staticmethod
    def positive_sample(label: np.ndarray):
        """Sample a positive sample from a label vector."""
        num_nonzero = np.count_nonzero(label)
        if num_nonzero == 0:
            return label  # no positive sample

        random_label = np.random.randint(0, 2, size=label.shape)
        positive_label = np.logical_and(label, random_label).astype(np.int32)
        if np.count_nonzero(positive_label) > 0:
            return positive_label
        else:
            return label

    @staticmethod
    def negative_sample(label: np.ndarray):
        """Sample a negative sample from a label vector."""
        random_label = np.random.randint(0, 2, size=label.shape)
        negative_label = np.logical_and(np.logical_not(label), random_label)
        negative_label = np.logical_or(
            negative_label, np.random.randint(0, 2, size=label.shape)
        ).astype(np.int32)
        return negative_label

    @staticmethod
    def translate(label: np.ndarray) -> str:
        """Translate label to string."""
        relation_str = []
        if label[0] > 0.5:
            relation_str.append("left")
        if label[1] > 0.5:
            relation_str.append("right")
        if label[2] > 0.5:
            relation_str.append("front")
        if label[3] > 0.5:
            relation_str.append("behind")
        if label[4] > 0.5:
            relation_str.append("on")
        if label[5] > 0.5:
            relation_str.append("above")
        if label[6] > 0.5:
            relation_str.append("in")
        if label[7] > 0.5:
            relation_str.append("under")
        if label[8] > 0.5:
            relation_str.append("near")
        if not relation_str:
            relation_str.append("none")
        return ",".join(relation_str)

    @staticmethod
    def vocabulary(label: np.ndarray) -> list[str]:
        """Translate label to string."""
        rel_sentence = []
        # possible relations
        if label[0] > 0.5:
            rel_sentence = []  # reset
            # left
            rel_sentence.append("A is left to B")
            if label[2] > 0.5:
                rel_sentence = []  # reset
                # front and left
                rel_sentence.append("A is at the front left of B")
                rel_sentence.append("A is in front left of B")
                rel_sentence.append("A is in left front of B")
                rel_sentence.append("A is left and front of B")
                if label[4] > 0.5:
                    rel_sentence = []  # reset
                    # on and left and front
                    rel_sentence.append("A is on left front side of B")
                    rel_sentence.append("A is on front left side of B")
                    rel_sentence.append("A is on left front corner of B")
                    rel_sentence.append("A is on front left corner of B")
                    rel_sentence.append("A is on and left and front of B")
                elif label[5] > 0.5:
                    rel_sentence = []
                    # above and left and front
                    rel_sentence.append("A is above left front side of B")
                    rel_sentence.append("A is above front left side of B")
                    rel_sentence.append("A is above left front corner of B")
                    rel_sentence.append("A is above front left corner of B")
                    rel_sentence.append("A is above and left and front of B")
                elif label[6] > 0.5:
                    rel_sentence = []
                    # in and left and front
                    rel_sentence.append("A is in left front side of B")
                    rel_sentence.append("A is in front left side of B")
                    rel_sentence.append("A is in left front corner of B")
                    rel_sentence.append("A is in front left corner of B")
                    rel_sentence.append("A is in and left and front of B")
                elif label[7] > 0.5:
                    rel_sentence = []
                    # under and left and front
                    rel_sentence.append("A is under left front side of B")
                    rel_sentence.append("A is under front left side of B")
                    rel_sentence.append("A is under left front corner of B")
                    rel_sentence.append("A is under front left corner of B")
                    rel_sentence.append("A is under and left and front of B")
            elif label[3] > 0.5:
                rel_sentence = []  # reset
                # behind and left
                rel_sentence.append("A is behind left of B")
                rel_sentence.append("A is left behind of B")
                rel_sentence.append("A is left and behind of B")
                rel_sentence.append("A is at the left back of B")
                if label[4] > 0.5:
                    rel_sentence = []  # reset
                    # on and left and behind
                    rel_sentence.append("A is on left back side of B")
                    rel_sentence.append("A is on back left side of B")
                    rel_sentence.append("A is on left back corner of B")
                    rel_sentence.append("A is on back left corner of B")
                    rel_sentence.append("A is on and left and back of B")
                elif label[5] > 0.5:
                    rel_sentence = []
                    # above and left and behind
                    rel_sentence.append("A is above left back side of B")
                    rel_sentence.append("A is above back left side of B")
                    rel_sentence.append("A is above left back corner of B")
                    rel_sentence.append("A is above back left corner of B")
                    rel_sentence.append("A is above and left and back of B")
                elif label[6] > 0.5:
                    rel_sentence = []
                    # in and left and behind
                    rel_sentence.append("A is in left back side of B")
                    rel_sentence.append("A is in back left side of B")
                    rel_sentence.append("A is in left back corner of B")
                    rel_sentence.append("A is in back left corner of B")
                    rel_sentence.append("A is in and left and back of B")
                elif label[7] > 0.5:
                    rel_sentence = []
                    # under and left and behind
                    rel_sentence.append("A is under left back side of B")
                    rel_sentence.append("A is under back left side of B")
                    rel_sentence.append("A is under left back corner of B")
                    rel_sentence.append("A is under back left corner of B")
                    rel_sentence.append("A is under and left and back of B")
            elif label[4] > 0.5:
                rel_sentence = []  # reset
                # on and left
                rel_sentence.append("A is on left side of B")
                rel_sentence.append("A is at left top of B")
                rel_sentence.append("A is on and left of B")
            elif label[5] > 0.5:
                rel_sentence = []  # reset
                # above and left
                rel_sentence.append("A is above left part of B")
                rel_sentence.append("A is above and left to B")
            elif label[6] > 0.5:
                rel_sentence = []  # reset
                # in and left
                rel_sentence.append("A is in left part of B")
                rel_sentence.append("A is in and left to B")
            elif label[7] > 0.5:
                rel_sentence = []  # reset
                # under and left
                rel_sentence.append("A is under left part of B")
                rel_sentence.append("A is under and left to B")
        elif label[1] > 0.5:
            rel_sentence = []  # reset
            # right
            rel_sentence.append("A is right to B")
            if label[2] > 0.5:
                rel_sentence = []  # reset
                # front and right
                rel_sentence.append("A is at the right front of B")
                rel_sentence.append("A is in front right of B")
                rel_sentence.append("A is in right front of B")
                rel_sentence.append("A is right and front of B")
                if label[4] > 0.5:
                    rel_sentence = []  # reset
                    # on and right and front
                    rel_sentence.append("A is on right front side of B")
                    rel_sentence.append("A is on front right side of B")
                    rel_sentence.append("A is on right front corner of B")
                    rel_sentence.append("A is on front right corner of B")
                    rel_sentence.append("A is on and right and front of B")
                elif label[5] > 0.5:
                    rel_sentence = []
                    # above and right and front
                    rel_sentence.append("A is above right front side of B")
                    rel_sentence.append("A is above front right side of B")
                    rel_sentence.append("A is above right front corner of B")
                    rel_sentence.append("A is above front right corner of B")
                    rel_sentence.append("A is above and right and front of B")
                elif label[6] > 0.5:
                    rel_sentence = []
                    # in and right and front
                    rel_sentence.append("A is in right front side of B")
                    rel_sentence.append("A is in front right side of B")
                    rel_sentence.append("A is in right front corner of B")
                    rel_sentence.append("A is in front right corner of B")
                    rel_sentence.append("A is in and right and front of B")
                elif label[7] > 0.5:
                    rel_sentence = []
                    # under and right and front
                    rel_sentence.append("A is under right front side of B")
                    rel_sentence.append("A is under front right side of B")
                    rel_sentence.append("A is under right front corner of B")
                    rel_sentence.append("A is under front right corner of B")
                    rel_sentence.append("A is under and right and front of B")
            elif label[3] > 0.5:
                rel_sentence = []  # reset
                # behind and right
                rel_sentence.append("A is at the right back of B")
                rel_sentence.append("A is behind right of B")
                rel_sentence.append("A is right behind of B")
                rel_sentence.append("A is right and behind of B")
                if label[4] > 0.5:
                    rel_sentence = []
                    # on and right and behind
                    rel_sentence.append("A is on right back side of B")
                    rel_sentence.append("A is on back right side of B")
                    rel_sentence.append("A is on right back corner of B")
                    rel_sentence.append("A is on back right corner of B")
                    rel_sentence.append("A is on and right and back of B")
                elif label[5] > 0.5:
                    rel_sentence = []
                    # above and right and behind
                    rel_sentence.append("A is above right back side of B")
                    rel_sentence.append("A is above back right side of B")
                    rel_sentence.append("A is above right back corner of B")
                    rel_sentence.append("A is above back right corner of B")
                    rel_sentence.append("A is above and right and back of B")
                elif label[6] > 0.5:
                    rel_sentence = []
                    # in and right and behind
                    rel_sentence.append("A is in right back side of B")
                    rel_sentence.append("A is in back right side of B")
                    rel_sentence.append("A is in right back corner of B")
                    rel_sentence.append("A is in back right corner of B")
                    rel_sentence.append("A is in and right and back of B")
                elif label[7] > 0.5:
                    rel_sentence = []
                    # under and right and behind
                    rel_sentence.append("A is under right back side of B")
                    rel_sentence.append("A is under back right side of B")
                    rel_sentence.append("A is under right back corner of B")
                    rel_sentence.append("A is under back right corner of B")
                    rel_sentence.append("A is under and right and back of B")
            elif label[4] > 0.5:
                rel_sentence = []  # reset
                # on and right
                rel_sentence.append("A is on right side of B")
                rel_sentence.append("A is at right top of B")
                rel_sentence.append("A is on and right of B")
            elif label[5] > 0.5:
                rel_sentence = []  # reset
                # above and right
                rel_sentence.append("A is above right part of B")
                rel_sentence.append("A is above and right to B")
            elif label[6] > 0.5:
                rel_sentence = []  # reset
                # in and right
                rel_sentence.append("A is in right part of B")
                rel_sentence.append("A is in and right to B")
            elif label[7] > 0.5:
                rel_sentence = []  # reset
                # under and right
                rel_sentence.append("A is under right part of B")
                rel_sentence.append("A is under and right to B")
        elif label[2] > 0.5:
            rel_sentence = []  # reset
            # front
            rel_sentence.append("A is in front of B")
            if label[4] > 0.5:
                rel_sentence = []  # reset
                # on and front
                rel_sentence.append("A is on front side of B")
                rel_sentence.append("A is at front top of B")
                rel_sentence.append("A is on and front of B")
            elif label[5] > 0.5:
                rel_sentence = []  # reset
                # above and front
                rel_sentence.append("A is above front part of B")
                rel_sentence.append("A is above and front to B")
            elif label[6] > 0.5:
                rel_sentence = []  # reset
                # in and front
                rel_sentence.append("A is in front part of B")
                rel_sentence.append("A is in and front to B")
            elif label[7] > 0.5:
                rel_sentence = []  # reset
                # under and front
                rel_sentence.append("A is under front part of B")
                rel_sentence.append("A is under and front to B")
        elif label[3] > 0.5:
            rel_sentence = []  # reset
            # behind
            rel_sentence.append("A is behind B")
            rel_sentence.append("A is in back of B")
            if label[4] > 0.5:
                rel_sentence = []  # reset
                # on and behind
                rel_sentence.append("A is on behind side of B")
                rel_sentence.append("A is at behind top of B")
                rel_sentence.append("A is on and behind of B")
            elif label[5] > 0.5:
                rel_sentence = []  # reset
                # above and behind
                rel_sentence.append("A is above behind part of B")
                rel_sentence.append("A is above and behind to B")
            elif label[6] > 0.5:
                rel_sentence = []  # reset
                # in and behind
                rel_sentence.append("A is in behind part of B")
                rel_sentence.append("A is in and behind to B")
            elif label[7] > 0.5:
                rel_sentence = []  # reset
                # under and behind
                rel_sentence.append("A is under behind part of B")
                rel_sentence.append("A is under and behind to B")
        elif label[4] > 0.5:
            rel_sentence = []  # reset
            # on
            rel_sentence.append("A is on B")
        elif label[5] > 0.5:
            rel_sentence = []  # reset
            # above
            rel_sentence.append("A is above B")
        elif label[6] > 0.5:
            rel_sentence = []  # reset
            # in
            rel_sentence.append("A is in B")
        elif label[7] > 0.5:
            rel_sentence = []  # reset
            # under
            rel_sentence.append("A is under B")
            rel_sentence.append("A is below B")
        # if is near, near doesn't conflict with other relations
        elif label[8] > 0.5:
            rel_sentence = []  # reset
            rel_sentence.append("A is near B")

        # filter out impossible relations
        if label[0] > 0.5 and label[1] > 0.5:
            rel_sentence = ["Impossible"]  # Can't be left and right
        if label[2] > 0.5 and label[3] > 0.5:
            rel_sentence = ["Impossible"]  # Can't be front and behind
        if label[4] > 0.5 and label[7] > 0.5:
            rel_sentence = ["Impossible"]  # Can't be  on and under
        if label[5] > 0.5 and label[6] > 0.5:
            rel_sentence = ["Impossible"]  # Can't be above and in
        if label[5] > 0.5 and label[7] > 0.5:
            rel_sentence = ["Impossible"]  # Can't be above and under
        if label[6] > 0.5 and label[7] > 0.5:
            rel_sentence = ["Impossible"]  # Can't be in and under

        if not rel_sentence:
            rel_sentence.append("A has no relationship with B")
        return rel_sentence

    @staticmethod
    def vocabulary_map():
        """Generate the vocabulary for all possible relation combination"""
        rel_map = {}
        for i in range(1 << NUM_BASE_REL):
            # transfer i to label
            binary_string = bin(i)[2:]
            label = [int(bit) for bit in binary_string]
            label = [0] * (NUM_BASE_REL - len(label)) + label
            rel_map[tuple(label)] = Points9.vocabulary(label)
        return rel_map


class SpatialRelSampler:
    """Sample spatial relation from a given pose"""

    def __init__(self, device):
        # build the basic rel vocabulary
        pose2rel_map = Points9.vocabulary_map()
        # check redundancy
        self.rel_list = []
        self.label_list = []
        for key, value in pose2rel_map.items():
            if value[0] not in self.rel_list:
                self.rel_list += value
            if value == ["Impossible"]:
                self.label_list.append((0, 0, 0, 0, 0, 0, 0, 0, 0))
            else:
                self.label_list.append(key)
        # remove duplicate
        self.rel_list = list(set(self.rel_list))
        # create P&N mask
        label_array = np.array(self.label_list)
        cross_or_sum = np.sum(
            np.logical_or(label_array[:, np.newaxis, :], label_array[np.newaxis, :, :]), axis=2
        )
        corss_and_sum = np.sum(
            np.logical_and(label_array[:, np.newaxis, :], label_array[np.newaxis, :, :]), axis=2
        )
        self_sum = np.sum(label_array, axis=1)
        mask = (cross_or_sum - self_sum) == 0
        mask[corss_and_sum == 0] = False
        mask[np.logical_and(corss_and_sum == 0, cross_or_sum == 0)] = True
        mask = mask.T.astype(np.float32)

        # combine rel
        self.full_rel_list = []
        for i in range(mask.shape[0]):
            full_rel_i = []
            for j in range(mask.shape[1]):
                if mask[i, j] == 1:
                    full_rel_i += pose2rel_map[self.label_list[j]]
            # remove duplicates
            full_rel_i = list(set(full_rel_i))
            self.full_rel_list.append(full_rel_i)

        # encode rel
        model, preprocess = clip.load("ViT-B/32", device=device)
        def encode(x): return model.encode_text(clip.tokenize(x).to(device)).detach().cpu().numpy()
        self.rel_embedding_map = {rel: encode(rel) for rel in self.rel_list}

    def sample_rel(self, label):
        """Sample a relation from a given pose"""
        label_idx = self.label_list.index(label)
        positive_rel = random.choice(self.full_rel_list[label_idx])
        #
        unrelevant_rel = [rel for rel in self.rel_list if rel not in self.full_rel_list[label_idx]]
        negative_rel = random.choice(unrelevant_rel)
        return positive_rel, negative_rel

    def sample_rel_embedding(self, label):
        """Sample a relation from a given pose"""
        label_idx = self.label_list.index(tuple(label))
        positive_rel = random.choice(self.full_rel_list[label_idx])
        #
        unrelevant_rel = [rel for rel in self.rel_list if rel not in self.full_rel_list[label_idx]]
        negative_rel = random.choice(unrelevant_rel)
        return self.rel_embedding_map[positive_rel], self.rel_embedding_map[negative_rel]
