import os
import cv2
import copy
import pickle
import argparse
from functools import partial
import PIL.Image as Image
import networkx
import numpy as np
import open3d as o3d
import clip
import torch
from scipy.spatial import distance
import pycocotools
from detectron2.structures import Instances
import warnings
from typing import List, Union


def read_detectron_instances(filepath: Union[str, os.PathLike], rle_to_mask=True) -> Instances:
    with open(filepath, "rb") as fp:
        instances = pickle.load(fp)
        if rle_to_mask:
            pred_masks = np.stack(
                [pycocotools.mask.decode(rle) for rle in instances.pred_masks_rle]
            )
            instances.pred_masks = torch.from_numpy(pred_masks)
    return instances


# class
class OVIMap:
    """Open vocabulary instance map"""

    def __init__(
        self, geometry_file, info_file, img_height, img_width, intrinsic, extrinsic, device
    ):
        # paths
        self.load_geometry(geometry_file, info_file)
        # data structure
        self.instances = []
        # camera params
        self.img_height = img_height
        self.img_width = img_width
        if extrinsic is None:
            center = np.asarray(self.vis_pcd.get_center())
            camera_pose = np.eye(4)
            camera_pose[:3, 3] = center + np.array([0, 0, self.scene_scale]) * 1.5
            # rotate along x-axis
            camera_pose[:3, 0] = np.array([1, 0, 0])
            camera_pose[:3, 1] = np.array([0, -1, 0])
            camera_pose[:3, 2] = np.array([0, 0, -1])
            extrinsic = camera_pose
        self.camera_params = o3d.camera.PinholeCameraParameters()
        self.update_camera(intrinsic, extrinsic)

        # clip model
        self.device = device
        self.clip_model, self.clip_preprocess = clip.load("ViT-B/32", self.device)
        # external geometry
        self.external_geometry = {}
        self.external_image = []
        self.external_color = []

    def update_camera(self, intrinsic, extrinsic):
        """Update camera parameters."""
        if intrinsic is not None:
            self.intrinsic = intrinsic
            intrinsic_params = o3d.camera.PinholeCameraIntrinsic(
                self.img_width,
                self.img_height,
                intrinsic[0, 0],
                intrinsic[1, 1],
                intrinsic[0, 2],
                intrinsic[1, 2],
            )
            self.camera_params.intrinsic = intrinsic_params
        if extrinsic is not None:
            self.extrinsic = extrinsic
            self.camera_params.extrinsic = self.extrinsic

    def load_geometry(self, geometry_file, info_file=None):
        """Load geometry from file"""
        if info_file:
            # load transfromation
            transformation = np.eye(4)
            try:
                if info_file.endswith("axis_alignment.txt"):
                    transformation = np.loadtxt(info_file)
                else:
                    with open(info_file, "r") as fp:
                        for line in fp.readlines():
                            line = line.strip()
                            if line.startswith("axisAlignment"):
                                transformation = (
                                    np.array(line.split(" = ")[1].split()).astype(float).reshape((4, 4))
                                )
                                break
            except:
                warnings.warn("Cannot load transformation.")
            self.scene_pcd = o3d.io.read_point_cloud(geometry_file)
            self.scene_pcd.transform(transformation)
            # update scene scale
            self.scene_scale = np.linalg.norm(
                np.asarray(self.scene_pcd.get_max_bound())
                - np.asarray(self.scene_pcd.get_min_bound())
            )
            self.vis_pcd = copy.deepcopy(self.scene_pcd)
        else:
            self.scene_pcd = o3d.io.read_point_cloud(geometry_file)
            # update scene scale
            self.scene_scale = np.linalg.norm(
                np.asarray(self.scene_pcd.get_max_bound())
                - np.asarray(self.scene_pcd.get_min_bound())
            )
            self.vis_pcd = copy.deepcopy(self.scene_pcd)

    def query(self, query, top_k=1):
        """Query the map with a query string."""
        top_k = min(top_k, len(self.instances))
        if isinstance(query, str):
            text_feature = self.get_clip_feature(query, normalize=True)
            text_feature = text_feature.cpu().detach().numpy().flatten()

            # compute feature distances
            distances = []
            for i, instance in enumerate(self.instances):
                instance_feature = self.get_feature(i)
                if instance_feature is not None:
                    # For ground-truth we don't have detic feature
                    feature_distance = distance.cosine(text_feature, instance_feature)
                    distances.append(feature_distance)
            distances = np.array(distances)
            ascending_indices = distances.argsort()
            matched_id = ascending_indices[:top_k]
            return matched_id.tolist()
        else:
            raise ValueError("Query type not supported.")

    # utils
    def get_pcd(self, instance_id):
        """Get the point cloud of an instance."""
        indices_3d = self.instances[instance_id]["pt_indices"]
        instance_pcd = copy.deepcopy(self.scene_pcd.select_by_index(indices_3d))
        return instance_pcd

    def get_img(self, instance_id) -> Union[np.ndarray, None]:
        """Get the image of an instance."""
        # return a white image
        return np.ones((self.img_height, self.img_width, 3), dtype=np.uint8) * 255

    def get_category(self, instance_id) -> List[str]:
        """Get the category of an instance."""
        return self.instances[instance_id]["top5_vocabs"]

    def get_feature(self, instance_id) -> Union[np.ndarray, None]:
        """Get the feature of an instance."""
        if (
            "feature" not in self.instances[instance_id]
            or self.instances[instance_id]["feature"] is None
        ):
            if self.instances[instance_id]["top5_vocabs"] is not None:
                # If we don't have image, we check category
                top_vocab = self.instances[instance_id]["top5_vocabs"][0]
                feature = self.get_clip_feature(top_vocab, normalize=True)
                feature = feature.cpu().detach().numpy().flatten()
                return feature
            else:
                return None
        else:
            feature = self.instances[instance_id]["feature"]
            return feature / np.linalg.norm(feature)

    def mark(self, instance_id: Union[List[int], int], color=None):
        """Mark an instance (instances) with a color."""
        if isinstance(instance_id, int):
            instance_id = [instance_id]
        for id in instance_id:
            self.external_image.append(self.get_img(id).astype(np.uint8))
            indices_3d = self.instances[id]["pt_indices"]
            pcd_colors = np.asarray(self.vis_pcd.colors)
            if color is None:
                # random color
                pcd_color = np.random.rand(3)
                pcd_color = pcd_color * 0.5 + 0.5
                pcd_colors[indices_3d] = pcd_color
            else:
                pcd_color = np.array(color)
                pcd_colors[indices_3d] = pcd_color
            self.external_color.append(pcd_color)

    def clear_mark(self):
        """Clear all marked instances."""
        self.external_image = []
        self.external_color = []
        self.vis_pcd.colors = copy.deepcopy(self.scene_pcd.colors)

    def draw_path(self, vis_image, path, color=None):
        """Draw a path on an image."""
        from utils.misc_utils import project_3d_points_to_2d

        if color is None:
            color = np.random.rand(3)
        # project path to image plane
        path_proj = project_3d_points_to_2d(np.vstack(path), self.extrinsic, self.intrinsic).astype(
            np.int32
        )
        for i in range(len(path_proj) - 1):
            # cv2.line(vis_image, tuple(path_proj[i]), tuple(path_proj[i + 1]), color, 2)
            cv2.circle(vis_image, tuple(path_proj[i]), 3, color, 1)
        return vis_image

    def reset(self):
        """Reset the ovimap."""
        # reset vis_pcd
        self.external_image = []
        self.external_color = []
        self.external_geometry = {}
        self.vis_pcd = copy.deepcopy(self.scene_pcd)

    def get_clip_feature(self, _input: Union[str, List[str], np.ndarray], normalize=True):
        """Get the clip feature of a text label."""
        if isinstance(_input, str):
            # text
            text_inputs = clip.tokenize(f"a {_input}").to(self.device)
            features = self.clip_model.encode_text(text_inputs)
        elif isinstance(_input, List):
            # multiple texts
            text_inputs = torch.cat([clip.tokenize(f"a {c}") for c in _input]).to(self.device)
            features = self.clip_model.encode_text(text_inputs)
        elif isinstance(_input, np.ndarray):
            image = Image.fromarray(_input)
            # image
            image_inputs = self.clip_preprocess(image).unsqueeze(0).to(self.device)
            features = self.clip_model.encode_image(image_inputs)
        if normalize:
            features /= features.norm(dim=-1, keepdim=True)
        return features

    def visualize_3d(self, **kwargs):
        """Show the scene with marked instances in 3d."""
        show_origin = kwargs.get("show_origin", False)
        vis_list = [self.vis_pcd]
        if show_origin:
            origin = o3d.geometry.TriangleMesh.create_coordinate_frame(
                size=0.1 * self.scene_scale, origin=[0, 0, 0]
            )
            vis_list.append(origin)
        # show extrinsic
        show_extrinsic = kwargs.get("show_extrinsic", False)
        if show_extrinsic:
            extrinsic = o3d.geometry.TriangleMesh.create_coordinate_frame(
                size=0.1 * self.scene_scale, origin=[0, 0, 0]
            )
            extrinsic.transform(np.linalg.inv(self.extrinsic))
            vis_list.append(extrinsic)
        # show bbox
        show_bbox = kwargs.get("show_bbox", False)
        if show_bbox:
            # nodes
            for instance_id in range(len(self.instances)):
                instance_pcd = self.get_pcd(instance_id)
                bbox = instance_pcd.get_axis_aligned_bounding_box()
                # create a ball for center
                radius = np.linalg.norm(bbox.get_max_bound() - bbox.get_min_bound())
                radius = 0.05 * np.sqrt(radius)
                instance_points = np.asarray(instance_pcd.points)
                instance_avg_position = np.mean(instance_points, axis=0)
                center_ball = o3d.geometry.TriangleMesh.create_sphere(radius=radius)
                center_ball.translate(instance_avg_position)
                # get random color
                color = np.random.rand(3)
                color = color * 0.5 + 0.5
                bbox.color = color
                center_ball.paint_uniform_color(color)
                self.external_geometry[f"{instance_id}_bbox"] = bbox
                self.external_geometry[f"{instance_id}_center"] = center_ballß

        # show external geometries
        external_geometry = kwargs.get("external_geometry", {})
        o3d.visualization.draw_geometries(
            [*vis_list, *self.external_geometry.values(), *external_geometry.values()]
        )

    def get_camera_image_top(self, **kwargs):
        """Get the top view of the scene."""
        lr_flip = kwargs.get("lr_flip", True)
        # set the scene with the point cloud
        vis = o3d.visualization.VisualizerWithKeyCallback()
        vis.create_window(width=self.img_width, height=self.img_height, visible=False)
        vis.get_render_option().point_size = 2
        vis.add_geometry(self.vis_pcd)
        for geometry in self.external_geometry.values():
            vis.add_geometry(geometry)
        # add the origin
        show_origin = kwargs.get("show_origin", True)
        if show_origin:
            origin = o3d.geometry.TriangleMesh.create_coordinate_frame(
                size=0.1 * self.scene_scale, origin=[0, 0, 0]
            )
            # shift the origin to the center of the scene
            center = self.scene_pcd.get_center()
            origin.translate(center)
            vis.add_geometry(origin)

        ctrl = vis.get_view_control()
        if self.camera_params is not None:
            ctrl.convert_from_pinhole_camera_parameters(self.camera_params, allow_arbitrary=True)

        # capture the image and close the visualizer
        image = vis.capture_screen_float_buffer(do_render=True)
        depth_image = vis.capture_depth_float_buffer(do_render=True)  # Capture depth buffer
        vis.destroy_window()

        # convert the image to a numpy array
        image = np.asarray(image)
        # convert the image from BGR to RGB
        image = image[:, :, [2, 1, 0]]

        # flip image, depth image along x axis
        if lr_flip:
            image = np.fliplr(image)
            depth_image = np.fliplr(depth_image)
        # change image to uint8
        image = (image * 255).astype(np.uint8)
        return image.copy()  # return a copy of the image, because this can be a view

    def render(self, **kwargs) -> cv2.UMat:
        """Render the scene with marked instances"""
        # base image
        bk_img = self.get_camera_image_top(**kwargs)
        # append instance image on right side
        x_offset = 0
        y_offset = 0
        # scale limit
        scale_x_lim = 0.15

        # instance images
        show_inst_img = kwargs.get("show_inst_img", False)
        if show_inst_img:
            for ext_img, ext_color in zip(self.external_image, self.external_color):
                # convert to BGR
                ext_color = (
                    int(ext_color[2] * 255),
                    int(ext_color[1] * 255),
                    int(ext_color[0] * 255),
                )
                # calculate the new size of the smaller image, keep ratio
                width = int(bk_img.shape[1] * scale_x_lim)
                height = int(ext_img.shape[0] * width / ext_img.shape[1])
                ext_img_resize = cv2.resize(ext_img, (width, height))
                # add a border of ext_color to ext_img
                border_size = 5
                new_height = height - (2 * border_size)
                new_width = width - (2 * border_size)
                img_no_borders = ext_img_resize[
                    border_size: new_height + border_size, border_size: new_width + border_size
                ]
                ext_img_resize = cv2.copyMakeBorder(
                    img_no_borders,
                    border_size,
                    border_size,
                    border_size,
                    border_size,
                    cv2.BORDER_CONSTANT,
                    value=ext_color,
                )
                # create a region of interest (ROI) in the larger image
                rows, cols, channels = ext_img_resize.shape
                # add the blended image to the larger image
                if (y_offset + rows) > bk_img.shape[0] or (x_offset + cols) > bk_img.shape[1]:
                    break

                roi = bk_img[y_offset: y_offset + rows, x_offset: x_offset + cols]
                # blend the two images
                alpha = 1.0  # Blending ratio (0 <= alpha <= 1)
                beta = 1.0 - alpha
                blended_image = cv2.addWeighted(ext_img_resize, alpha, roi, beta, 0.0)
                bk_img[y_offset: y_offset + rows, x_offset: x_offset + cols] = blended_image

                # update offset
                y_offset += height

        return bk_img

    def get_top_image_info(self, lr_flip: bool, **kwargs):
        """return top image, intrinsic, extrinsic"""
        image = self.get_camera_image_top(lr_flip=lr_flip, **kwargs)
        return image, self.intrinsic, self.extrinsic

    def save_top_image_info(self, save_path: str, lr_flip: bool, **kwargs):
        """save top image, intrinsic, extrinsic"""
        image = self.get_camera_image_top(lr_flip=lr_flip)
        # save image
        cv2.imwrite(save_path, image)
        # save intrinsic and extrinsic
        np.savez(
            save_path.replace(".png", ".npz"), intrinsic=self.intrinsic, extrinsic=self.extrinsic
        )


class OVIMapDetic(OVIMap):
    """Open vocabulary instance map based on 3d detic feature."""

    def __init__(
        self,
        geometry_file,
        info_file,
        img_height,
        img_width,
        intrinsic,
        extrinsic,
        device,
        detic_path,
        anno_file,
        color_img_path,
    ):
        super().__init__(
            geometry_file, info_file, img_height, img_width, intrinsic, extrinsic, device
        )
        # path related
        self.detic_path = detic_path
        self.color_img_path = color_img_path
        # load instances
        with open(anno_file, "rb") as fp:
            self.instances = pickle.load(fp)
            if type(self.instances) == networkx.classes.digraph.DiGraph:
                instance_nodes = self.instances.nodes(data=True)
                self.instances = [
                    {
                        "instance_id": node[0],
                        "feature": node[1]["feature"],
                        "pt_indices": node[1]["pt_indices"],
                        "top5_vocabs": node[1]["top5_vocabs"],
                        "detections": node[1]["detections"],
                    }
                    for node in instance_nodes
                ]

    # Interface
    def get_img(self, instance_id) -> Union[np.ndarray, None]:
        """Get the image of an instance, detic-fusion format."""
        if (
            "image" in self.instances[instance_id]
            and self.instances[instance_id]["image"] is not None
        ):
            return self.instances[instance_id]["image"]
        elif (
            "detections" in self.instances[instance_id]
            and self.instances[instance_id]["detections"] is not None
        ):
            (frame_id, detection_id, _) = self.instances[instance_id]["detections"][0]
            detic_result_path = os.path.join(self.detic_path, "instances", f"{frame_id}-color.pkl")
            detic_output = read_detectron_instances(detic_result_path)
            pred_scores = detic_output.scores.numpy()  # (M,)
            pred_masks = detic_output.pred_masks.numpy()  # (M, H, W)
            im_path = os.path.join(self.color_img_path, f"{frame_id}-color.jpg")
            # img = cv2.cvtColor(cv2.imread(im_path), cv2.COLOR_BGR2RGB)
            img = cv2.imread(im_path)
            mask = pred_masks[detection_id].astype(bool)
            # img[~mask] = 0
            return img
        else:
            # return a white image
            return super().get_img(instance_id)

    # Utils
    @classmethod
    def parse_path(cls, data_path, scene_name, annotation_file, dataset="scannet", **kwargs):
        """Parse needed file for detic ovi map"""
        detic_exp = kwargs.get("detic_exp", None)
        if dataset == "scannet":
            scene_path = os.path.join(data_path, "aligned_scans", scene_name)
            geometry_file = os.path.join(scene_path, f"{scene_name}_vh_clean_2.ply")
            detic_path = os.path.join(scene_path, "detic_output", detic_exp)
            anno_file = os.path.join(
                scene_path, "detic_output", detic_exp, "predictions", annotation_file
            )
            info_file = os.path.join(scene_path, f"{scene_name}.txt")
            color_img_path = os.path.join(scene_path, "color")
        elif dataset == "custom_scannet":
            scene_path = os.path.join(data_path, scene_name)
            geometry_file = os.path.join(scene_path, f"{scene_name}_vh_clean_2.ply")
            detic_path = os.path.join(scene_path, "detic_output", detic_exp)
            anno_file = os.path.join(
                scene_path, "detic_output", detic_exp, "predictions", annotation_file
            )
            info_file = os.path.join(scene_path, "axis_alignment.txt")
            color_img_path = os.path.join(scene_path, "color")
        elif dataset == "simple_fusion":
            scene_path = os.path.join(data_path, scene_name)
            geometry_file = os.path.join(scene_path, "recon.pcd")
            anno_file = os.path.join(scene_path, annotation_file)
            info_file = None
            detic_path = None
            color_img_path = os.path.join(scene_path, "color")
        return geometry_file, anno_file, info_file, color_img_path, detic_path


if __name__ == "__main__":
    # test case 1
    root_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
    data_path = f"{root_dir}/test_data"
    scene_name = "cus_scene0001_00"
    detic_exp = "imagenet21k-0.3"
    annotation_file = "proposed_fusion_detic_iou-0.25_recall-0.50_feature-0.75_interval-300.pkl"
    geometry_file, anno_file, info_file, color_img_path, detic_path = OVIMapDetic.parse_path(
        data_path, scene_name, annotation_file, "custom_scannet", detic_exp=detic_exp
    )
    intrinsic = np.array([[1000, 0, 399.5], [0, 1000, 499.5]])  # o3d requires .5
    img_height, img_width = 800, 1000
    ovimap = OVIMapDetic(
        geometry_file,
        info_file,
        img_height,
        img_width,
        intrinsic,
        None,
        device="cpu",
        detic_path=detic_path,
        anno_file=anno_file,
        color_img_path=color_img_path,
    )

    # Querie No.1
    queried_instance = ovimap.query("toy", top_k=10)
    for i, inst in enumerate(queried_instance):
        ovimap.mark(inst)
    # img = ovimap.render(show_inst_img=True)
    ovimap.visualize_3d()
    ovimap.clear_mark()

    # Querie No.2
    queried_instance = ovimap.query("trash can", top_k=3)
    for i, inst in enumerate(queried_instance):
        ovimap.mark(inst)
    # img = ovimap.render(show_inst_img=True)
    ovimap.visualize_3d()
    ovimap.clear_mark()

    # Querie No.3
    queried_instance = ovimap.query("drawer", top_k=10)
    for i, inst in enumerate(queried_instance):
        ovimap.mark(inst)
    # img = ovimap.render(show_inst_img=True)
    ovimap.visualize_3d()
    ovimap.clear_mark()
