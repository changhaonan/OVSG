""" Simple fusion method based grounding SAM"""
import open3d as o3d
import numpy as np
import os
import cv2
import pickle


def create_pcd(color_image, depth_image, intrinsic, extrinsic, masks, **kwargs):
    depth_scale = kwargs.get("depth_scale", 1.0)
    depth_trunc = kwargs.get("depth_trunc", 1.0)
    """Create point cloud from color and depth images"""
    intrinsic_o3d = o3d.camera.PinholeCameraIntrinsic(
        color_image.shape[1],  # width
        depth_image.shape[0],  # height
        intrinsic[0][0],  # fx
        intrinsic[1][1],  # fy
        intrinsic[0][2],  # cx
        intrinsic[1][2],  # cy
    )

    point_cloud_list = []
    bbox_list = []
    for i in range(masks.shape[0]):
        depth_image_masked = depth_image * masks[i]
        rgbd_image = o3d.geometry.RGBDImage.create_from_color_and_depth(
            o3d.geometry.Image(color_image),
            o3d.geometry.Image(depth_image_masked),
            depth_trunc=depth_trunc,  # truncate depth values greater than 3 meters
            convert_rgb_to_intensity=False,  # don't convert RGB to intensity
            depth_scale=depth_scale,  # depth values are in millimeters
        )

        point_cloud = o3d.geometry.PointCloud.create_from_rgbd_image(rgbd_image, intrinsic_o3d)
        point_cloud.transform(extrinsic)
        point_cloud_list.append(point_cloud)
        # create bbox
        bbox = o3d.geometry.AxisAlignedBoundingBox.create_from_points(point_cloud.points)
        bbox.color = (0, 1, 0)
        bbox_list.append(bbox)
    return point_cloud_list, bbox_list


def parse_single_frame(frame_path):
    # get point cloud
    # list files
    extrinsic = np.loadtxt(os.path.join(frame_path, "extrinsics.txt"))
    intrinsic = np.loadtxt(os.path.join(frame_path, "intrinsics.txt"))

    # read the output from grounding SAM
    sam_detection = pickle.load(open(os.path.join(frame_path, "grounded_sam_detections.pkl"), "rb"))

    # load image
    color_image = cv2.imread(os.path.join(frame_path, "color.png"))
    color_image = cv2.cvtColor(color_image, cv2.COLOR_BGR2RGB)
    depth_image = np.load(os.path.join(frame_path, "depth.npy"))
    if depth_image.dtype == np.uint16:
        depth_image = depth_image.astype(np.float32) / 1000.0
    depth_image = depth_image.astype(np.float32)

    extrinsic[:3, 3] /= 1000.0
    # create point cloud
    pcd_list, bbox_list = create_pcd(
        color_image, depth_image, intrinsic, extrinsic, sam_detection["mask"]
    )

    return color_image, depth_image, extrinsic, intrinsic, pcd_list


def main():
    color_image, depth_image, extrinsic, intrinsic, pcd_list = parse_single_frame(
        "/home/robot-learning/Projects/CDR/data/RGBD_intrinsics_extrinsics/detection"
    )
    o3d.visualization.draw_geometries(pcd_list)
    # compute stochastics
    pcd_center = np.array(pcd_list[0].get_center())
    print(pcd_center)


if __name__ == "__main__":
    main()
