from typing import List, Union, Dict, Any, Tuple
import pickle
import open3d as o3d
import cv2
import torch
from PIL import Image
import numpy as np
import tqdm
import random
import os
from ovsg.env.ovimap.ovimap import OVIMapDetic
from ovsg.env.algo.region import Region2D
from ovsg.env.algo.notion import (
    Space,
    Notion,
    Domain,
    Feature,
    FeatureType,
)
from ovsg.env.algo.notion_utils import parse_space_domain
from ovsg.env.notiondb import NotionDB
from ovsg.utils.spatial.spatial import Points9 as p9


class NotionOVIDB(NotionDB):
    """Notion DataBase equipped with OVIMapDetic"""

    scene_map: OVIMapDetic

    def __init__(self, cfg):
        super().__init__(cfg)
        self.scene_map = None
        # parse scene related info
        if cfg.render_intrinsic is None or cfg.render_intrinsic == "None":
            intrinsic = None
        else:
            intrinsic = np.array(cfg.render_intrinsic)
        if cfg.render_extrinsic is None or cfg.render_extrinsic == "None":
            extrinsic = None
        else:
            extrinsic = np.array(cfg.render_extrinsic)
        self.scene_map_cfg = {
            "data_path": cfg.ovi_data_path,
            "scene_name": cfg.ovi_scene_name,
            "detic_exp": cfg.ovi_detic_exp,
            "annotation_file": cfg.ovi_annotation_file,
            "annotation_gt_file": cfg.ovi_annotation_gt_file,
            "dataset": cfg.ovi_dataset,
            "device": cfg.notion_device,
            "intrinsic": intrinsic,
            "extrinsic": extrinsic,
            "img_width": cfg.render_width,
            "img_height": cfg.render_height,
        }
        self.console_text = ""
        self.motion_plans = []
        self.regions = []
        self.region_resolution = cfg.region_resolution
        # flag
        self.is_init = False
        # log
        self.exp_name = cfg.exp_name
        # eval param
        self.eval_top_k = cfg.eval_top_k
        self.eval_iou_thresh = cfg.eval_iou_thresh
        self.eval_enable_user = cfg.eval_enable_user

    def reset(self):
        """Reset the system"""
        # reset ovi map
        if self.scene_map is not None:
            self.scene_map.reset()
        # reset flag
        self.is_init = False
        # reset rest
        return super().reset()

    # Interface
    def search(
        self,
        query: str,
        top_k: int,
        domain: Union[Domain, None] = None,
        space: Union[Space, None] = None,
        **kwargs,
    ):
        """Generic Search method"""
        return self.notion_graph.search(
            query=query, top_k=top_k, domain=domain, space=space, **kwargs
        )

    def search_instance(self, query: str, top_k: int, **kwargs):
        """Search instance"""
        return self.search(query=query, top_k=top_k, domain=Domain.INS, **kwargs)

    def search_region(self, query: str, top_k: int, **kwargs):
        """Search region"""
        return self.search(query=query, top_k=top_k, domain=Domain.RGN, **kwargs)

    def search_tag(self, query: str, top_k: int, **kwargs):
        """Search text"""
        return self.search(query=query, top_k=top_k, domain=Domain.TAG, **kwargs)

    def search_user(self, query: str, top_k: int, **kwargs):
        """Search user"""
        return self.search(query=query, top_k=top_k, domain=Domain.USR, **kwargs)

    # Render related method
    def render(self, notions: Union[List[Notion], Notion, None] = None, title=""):
        """Render the system"""
        if notions is not None:
            if isinstance(notions, Notion):
                notions = [notions]
            self.scene_map.reset()
            for notion in notions:
                # for instance
                if notion.domain == Domain.INS and notion.space == Space.STATIC:
                    self.scene_map.mark(instance_id=notion.attributes["instance_id"])
                # for user
                elif notion.domain == Domain.USR:
                    color = np.random.rand(3)
                    color = color * 0.5 + 0.5
                    self.scene_map.external_image.append(notion.content.photo)
                    self.scene_map.external_color.append(color)
                    # add sphere represent user
                    sphere = o3d.geometry.TriangleMesh.create_sphere(radius=0.5)
                    sphere.compute_vertex_normals()
                    sphere.paint_uniform_color(color)
                    sphere.translate(notion.address[:3])
                    self.scene_map.external_geometry[notion.name] = sphere
                # for region
                elif notion.domain == Domain.RGN and notion.space == Space.REGION:
                    # color = np.random.rand(3)
                    # color = color * 0.5 + 0.5
                    # pcd = o3d.geometry.PointCloud()
                    # pcd.points = o3d.utility.Vector3dVector(notion.content.grid_points_3d())
                    # pcd.paint_uniform_color(color)
                    # self.scene_map.external_geometry[notion.name] = pcd
                    pass

        image = cv2.UMat(self.scene_map.render(show_inst_img=True, lr_flip=False))

        cv2.imshow(title, image)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

    def render_region(self, region: str):
        """Render region"""
        notion_region = self.search(query=region, top_k=1, domain=Domain.RGN)[0]
        notions_in_region = []
        for notion in self.notion_graph.notions:
            if (
                notion.domain == Domain.INS
                and notion.space == Space.STATIC
                and notion_region.space == Space.REGION
            ):
                if notion.address in notion_region.content:
                    notions_in_region.append(notion)
        self.render(notions=notions_in_region, title=region)

    def render_links(self, notion: Notion, relation: Union[str, None] = None):
        """Render notions linked to the notion"""
        notions_linked = []

        if relation is not None:
            # forward links
            for linked_id in notion.forward_links:
                linked_notion = self.notion_graph.notions[linked_id]
                if linked_notion.domain == Domain.INS and linked_notion.space == Space.STATIC:
                    link = self.notion_graph.links[(notion.id, linked_notion.id)]
                    if relation in link.relation_desc:
                        notions_linked.append(linked_notion)
            # backward links
            for linked_id in notion.backward_links:
                linked_notion = self.notion_graph.notions[linked_id]
                if linked_notion.domain == Domain.INS and linked_notion.space == Space.STATIC:
                    link = self.notion_graph.links[(linked_notion.id, notion.id)]
                    if relation in link.relation_desc:
                        notions_linked.append(linked_notion)
        else:
            # show all relations
            for linked_id in notion.links:
                linked_notion = self.notion_graph.notions[linked_id]
                if linked_notion.domain == Domain.INS and linked_notion.space == Space.STATIC:
                    notions_linked.append(linked_notion)
        self.render(notions=notions_linked, title=f"{notion.name}-{relation}")

    def render_spatial_links(self, notion: Notion, relation: str, A2B: bool = True):
        """Render spatial relations"""
        notions_linked = [notion]
        # debug
        pose_pair_list = []
        # encode relation
        query_relation = self.notion_graph.encode_text(relation, encoder="clip")
        # forward links
        if A2B:
            # A2B meaning A is the notion, B is the linked notion
            for linked_id in notion.forward_links:
                linked_notion = self.notion_graph.notions[linked_id]
                if linked_notion.space != Space.VIRTUAL:
                    # check spatial relationship
                    link = self.notion_graph.links[(notion.id, linked_notion.id)]
                    for relation_key, relation_desc in zip(link.relation_key, link.relation_desc):
                        if relation_desc == "spatial":
                            sim = self.notion_graph.generate("spatial_sprob", feature_1=query_relation, feature_2=relation_key)
                            pose_pair_list.append(
                                np.vstack([notion.address, linked_notion.address])
                            )
                            if sim > 0.3:
                                notions_linked.append(linked_notion)
                            # test
                            notion_address = notion.address
                            linked_notion_address = linked_notion.address
                            near = p9.is_near(notion_address, linked_notion_address)
                            if near:
                                print(f"near: {notion.name} - {linked_notion.name}")
        else:
            # B2A meaning B is the notion, A is the linked notion
            for linked_id in notion.backward_links:
                linked_notion = self.notion_graph.notions[linked_id]
                if linked_notion.space != Space.VIRTUAL:
                    # check spatial relationship
                    link = self.notion_graph.links[(linked_notion.id, notion.id)]
                    for relation_key, relation_desc in zip(link.relation_key, link.relation_desc):
                        if relation_desc == "spatial":
                            sim = self.notion_graph.generate("spatial_sprob", feature_1=query_relation, feature_2=relation_key)
                            pose_pair_list.append(
                                np.vstack([linked_notion.address, notion.address])
                            )
                            if sim > 0.3:
                                notions_linked.append(linked_notion)
        self.render(notions=notions_linked, title=f"{notion.name}-{relation}")
        return pose_pair_list

    def step_render(self, action=None):
        """Render one step"""
        self.render()

    # Visualize
    def visualize_3d(self, **kwargs):
        """visualize the scene graph in 3d"""
        # get links between instances
        external_geometry = {}
        relations = kwargs.get("relations", [])
        for link in self.notion_graph.links.values():
            notion1 = self.notion_graph.notions[link.id1]
            notion2 = self.notion_graph.notions[link.id2]
            if notion1.space is not Space.VIRTUAL and notion2.space is not Space.VIRTUAL:
                for relation in link.relation_desc:
                    if relation in relations:
                        # add node geometry
                        if f"bbox_{notion1.id}" not in external_geometry:
                            bbox, center = p9.to_geometry(notion1.address)
                            external_geometry[f"bbox_{notion1.id}"] = bbox
                            external_geometry[f"center_{notion1.id}"] = center
                        if f"bbox_{notion2.id}" not in external_geometry:
                            bbox, center = p9.to_geometry(notion2.address)
                            external_geometry[f"bbox_{notion2.id}"] = bbox
                            external_geometry[f"center_{notion2.id}"] = center
                        # add link geometry
                        if (notion1.id, notion2.id) not in external_geometry:
                            line = o3d.geometry.LineSet()
                            line.points = o3d.utility.Vector3dVector(
                                np.vstack([notion1.address[:3], notion2.address[:3]])
                            )
                            line.lines = o3d.utility.Vector2iVector(np.array([[0, 1]]))
                            # notion1 color
                            line.paint_uniform_color(external_geometry[f"bbox_{notion1.id}"].color)
                            external_geometry[(notion1.id, notion2.id)] = line
                        break
        kwargs["external_geometry"] = external_geometry

        self.scene_map.visualize_3d(**kwargs)

    def idle(self, idle_time=5):
        """Idle run to render"""
        for _ in range(idle_time):
            self.step_render()

    # Building method
    def build_from_ovi(self, use_gt_map: bool = False, enable_user: bool = False):
        """Build notion graph with ovi map"""
        # reset
        self.reset()
        # build scene map
        annotation_file = (
            self.scene_map_cfg["annotation_gt_file"]
            if use_gt_map
            else self.scene_map_cfg["annotation_file"]
        )

        geometry_file, anno_file, info_file, color_img_path, detic_path = OVIMapDetic.parse_path(
            self.scene_map_cfg["data_path"],
            self.scene_map_cfg["scene_name"],
            annotation_file,
            self.scene_map_cfg["dataset"],
            detic_exp=self.scene_map_cfg["detic_exp"],
        )
        extrinsic = self.scene_map_cfg["extrinsic"]
        intrinsic = self.scene_map_cfg["intrinsic"]
        img_height = self.scene_map_cfg["img_height"]
        img_width = self.scene_map_cfg["img_width"]
        self.scene_map = OVIMapDetic(
            geometry_file,
            info_file,
            img_height,
            img_width,
            intrinsic,
            extrinsic,
            device="cpu",
            detic_path=detic_path,
            anno_file=anno_file,
            color_img_path=color_img_path,
        )
        # build links
        self._build_from_ovi_wo_user()

        if enable_user:
            # add user
            luffy_photo = cv2.imread(os.path.join(self.notion_dir, "luffy.jpg"))
            luffy_pos = p9.from_bbox(pos=np.array([1.0, 0.0, 1.0]), size=np.array([0.5, 0.5, 2.0]))
            self.add_user(pos=luffy_pos, names=["luffy", "captain", "user"], image=luffy_photo)

            nami_photo = cv2.imread(os.path.join(self.notion_dir, "nami.jpg"))
            nami_pos = p9.from_bbox(pos=np.array([-1.0, 1.0, 1.0]), size=np.array([0.5, 0.5, 2.0]))
            self.add_user(pos=nami_pos, names=["nami", "navigator", "user"], image=nami_photo)

            zoro_photo = cv2.imread(os.path.join(self.notion_dir, "zoro.png"))
            zoro_pos = p9.from_bbox(pos=np.array([-1.0, -3.0, 1.0]), size=np.array([0.5, 0.5, 2.0]))
            self.add_user(pos=zoro_pos, names=["zoro", "sword", "user"], image=zoro_photo)

            if use_gt_map:
                # generate user instance relation from gt map
                relation_list = ["love", "like", "neutral", "dislike", "hate"]
                self._random_user_ins_relation(relation_list=relation_list)
            else:
                query_path = os.path.join(self.output_path(), "graph_queries")
                with open(os.path.join(query_path, "ovimap_gt.pkl"), "rb") as fp:
                    ovi_gt = pickle.load(fp)
                with open(
                    os.path.join(os.path.dirname(os.path.dirname(query_path)), "gt_match.pkl"), "rb"
                ) as fp:
                    gt_match = pickle.load(fp)
                for user_notion in self.notion_graph.at(domain=Domain.USR):
                    for ins_notion in self.notion_graph.at(domain=Domain.INS, space=Space.STATIC):
                        if ins_notion.id in gt_match["ovi2gt_map"]:
                            gt_id = gt_match["ovi2gt_map"][ins_notion.id][0]
                            for rel in ovi_gt[user_notion.name]:
                                if rel != "spatial":
                                    if gt_id in ovi_gt[user_notion.name][rel]:
                                        self.notion_graph.link(
                                            notion1=user_notion, relation=rel, notion2=ins_notion
                                        )
        # build notion
        self.notion_graph.update()

    def _build_from_ovi_wo_user(self):
        """Build notion nodes with ovi map without user"""
        tag_list = ["expensive", "cheap", "big", "small", "heavy", "light", "hard", "soft"]
        # build instance notion
        tqdm_desc = f"Building OVIMAP without user..."
        for i in tqdm.tqdm(range(len(self.scene_map.instances)), desc=tqdm_desc, unit='instances'):
            inst_pcd = self.scene_map.get_pcd(i)
            inst_img = self.scene_map.get_img(i)
            inst_cat = self.scene_map.get_category(i)[0]
            inst_feature = self.scene_map.get_feature(i)
            pos = p9.from_pcd(pcd=inst_pcd)
            random_tag = random.choice(tag_list)
            if inst_feature is not None:
                inst_feature = Feature(inst_feature, FeatureType.DETICIMG)
            else:
                inst_feature = self.notion_graph.encode_text(inst_cat, encoder="clip")
            self.add_notion(
                keys=[inst_feature],
                content=inst_pcd,
                pos=pos,
                image=inst_img,
                name=inst_cat,
                domain=Domain.INS,
                tags=[random_tag],
                space=Space.STATIC,
                instance_id=i,
            )
        # build regions notion
        scene_path = os.path.join(
            self.scene_map_cfg["data_path"], "aligned_scans", self.scene_map_cfg["scene_name"]
        )
        for region_name in self.regions:
            region_file = os.path.join(scene_path, f"{region_name}.npz")
            # region height is 3m by default
            region = Region2D(
                resolution=self.region_resolution,
                grid_size=[0, 0, int(3.0 / self.region_resolution)],
                name=region_name,
            )
            if os.path.exists(region_file) and not self.rebuild_region:
                region.load(region_file)
            else:
                top_image, intrinsic, extrinsic = self.scene_map.get_top_image_info(lr_flip=False)
                region.create_from_image(top_image, extrinsic, intrinsic, top_down=True)
                region.save(region_file)
            # add region notion
            region_feature = self.notion_graph.encode_text(text=region_name, encoder="clip")
            # can be queried by "region"
            region_names = [region_name]
            region_feature = [
                self.notion_graph.encode_text(text=region_name, encoder="clip")
                for region_name in region_names
            ]
            # parse pose
            region_pose = p9.from_pcd(pcd=region.grid_points_3d())
            self.add_notion(
                keys=region_feature,
                content=region,
                pos=region_pose,
                domain=Domain.RGN,
                name=region_name,
                space=Space.REGION,
            )

    def _random_user_ins_relation(self, relation_list: List[str]):
        """Generate Random user instance relation"""
        for user_notion in self.notion_graph.at(domain=Domain.USR):
            for ins_notion in self.notion_graph.at(domain=Domain.INS, space=Space.STATIC):
                # if random.random() > 0.5:
                if True:
                    relation = random.choice(relation_list)
                    self.notion_graph.link(
                        notion1=user_notion, relation=relation, notion2=ins_notion
                    )

    # Query method
    def _query_instance(
        self,
        query: str,
        top_k: int = 1,
        region: Union[str, None] = None,
        nearby: Union[List[str], None] = None,
    ):
        """Query instance"""
        # query ins
        queried_ins_notions, _ = self.notion_graph.search(
            query, top_k=top_k, domain=Domain.INS, space=Space.STATIC
        )
        queried_ins_notions_filter = queried_ins_notions
        if region:
            # query region
            queried_rgn_notion, __ = self.notion_graph.search(region, top_k=1, space=Space.REGION)[
                0
            ]
            queried_rgn_notion.view()
            # filter by region
            queried_ins_notions_filter = [
                ins_notion
                for ins_notion in queried_ins_notions
                if ins_notion.address in queried_rgn_notion.content
            ]
        if nearby:
            pass
        return queried_ins_notions, queried_ins_notions_filter

    # Query internface
    def query(self, query_str: str, top_k: int, method: str, verbose: bool = False):
        """Query API"""
        target_name, target_type, target_id = self.parse_syntax(self.notion_query, query_str)
        # match edges
        space, domain = parse_space_domain(type_str=target_type)
        # keywords
        if target_name.lower() in ["SOMEONE", "SOMEWHERE", "SOMETHING"]:
            top_k = -1

        if method == "prob":
            sprobs, candidates, matches, raw_candidates = self.graph_kernel_prob(
                notion_graph=self.notion_graph,
                notion_query=self.notion_query,
                target=target_name,
                top_k=top_k,
                domain=domain,
                space=space,
                verbose=verbose,
            )
        elif method in ["gnn", "jaccard", "head", "szymkiewicz_simpson"]:
            sprobs, candidates, matches, raw_candidates = self.graph_kernel(
                notion_graph=self.notion_graph,
                notion_query=self.notion_query,
                target=target_name,
                top_k=top_k,
                domain=domain,
                space=space,
                kernel_method=method,
                verbose=verbose,
            )
        else:
            raise ValueError(f"Unknown query method: {method}")
        return {
            "sprobs": sprobs,
            "candidates": candidates,
            "matches": matches,
            "raw_candidates": raw_candidates,
            "target_name": target_name,
            "target_id": target_id,
        }

    def query_and_render(self, query_str: str, top_k: int, method: str, verbose: bool, **kwargs):
        """Query and render"""
        result = self.query(query_str=query_str, top_k=top_k, method=method, verbose=verbose)
        print("@@@@@@@@@@@@@@@@")
        for _match, similarity in zip(result["matches"], result["sprobs"]):
            print(f"sim: {similarity}")
            for key, value in _match.items():
                key.view()
                value.view()
            print("------------------")

        # self.render(notions=result["candidates"], title="candidates")

        # get notions
        show_top_k = kwargs.get("show_top_k", 1)
        for i in range(show_top_k):
            notions_matched = []
            for key, value in result["matches"][i].items():
                notions_matched.append(self.notion_graph.notions[value.id1])
                notions_matched.append(self.notion_graph.notions[value.id2])
            notions_matched = list(set(notions_matched))
            # put the target first
            target = result["candidates"][i]
            if target not in notions_matched:
                notions_matched = [target] + notions_matched
            else:
                notions_matched.remove(target)
                notions_matched = [target] + notions_matched
            self.render(notions=notions_matched, title=f"top_{i}")
        self.scene_map.visualize_3d(show_origin=True)
        return result

    # main interface
    def chat(
        self,
        str_msg: str,
        img_msg: Union[List[Image.Image], List[np.ndarray], None] = None,
        **kwargs,
    ):
        """Chat"""
        # prepare
        if not self.is_init:
            self.build_from_ovi(use_gt_map=False, enable_user=False)
            # update flag
            self.is_init = True

        method = kwargs.get("query_method", "prob")
        enable_render = kwargs.get("enable_render", False)
        if enable_render:
            result = self.query_and_render(query_str=str_msg, top_k=10, method=method, verbose=False)
        else:
            result = self.query(query_str=str_msg, top_k=10, method=method, verbose=False)
        # organize output
        output = {
            "action": str_msg,
            "address": result["candidates"][0].address.tolist(),
        }
        return output, True

    # debug method
    def debug_pose(self):
        """Debug pose"""
        # Save instance 9-D pose for debug
        pose_list = []
        for ins_notion in self.notion_graph.at(domain=Domain.INS, space=Space.STATIC):
            pose_list.append(ins_notion.address)
        pose_list = np.array(pose_list)
        np.save(os.path.join(self.notion_dir, "instance_poses.npy"), pose_list)

    def debug(self):
        """Debug"""
        if not self.is_init:
            self.build_from_ovi(use_gt_map=False, enable_user=False)
            # update flag
            self.is_init = True

        sofa_candidate, _ = self.search_instance("wooden block", top_k=10)
        self.render_spatial_links(sofa_candidate[0], relation="A is near B", A2B=True)

    # Eval-related methods
    def output_path(self):
        """Output path"""
        root_data_path = self.scene_map_cfg["data_path"]
        with_user = "with" if self.eval_enable_user else "no"
        return os.path.join(
            root_data_path, self.exp_name, self.scene_map_cfg["scene_name"], with_user + "_user"
        )

    def gen_gt_match(self, iou_thresh: float = 0.5):
        """Generate gt match between gt and ovi"""
        # build gt map
        self.build_from_ovi(use_gt_map=True, enable_user=False)
        gt_notion_bbox = np.vstack(self.notion_graph.generate(target="notion_pos"))[:, 3:]
        gt_notion_name = [notion.name for notion in self.notion_graph.notions]
        self.build_from_ovi(use_gt_map=False, enable_user=False)
        ovi_notion_bbox = np.vstack(self.notion_graph.generate(target="notion_pos"))[:, 3:]
        ovi_notion_name = [notion.name for notion in self.notion_graph.notions]

        def bbox_iou(bbox_1, bbox_2):
            # compute the intersection of the bounding boxes
            x_min = np.maximum(bbox_1[:, 0].reshape(-1, 1), bbox_2[:, 0])
            y_min = np.maximum(bbox_1[:, 1].reshape(-1, 1), bbox_2[:, 1])
            z_min = np.maximum(bbox_1[:, 2].reshape(-1, 1), bbox_2[:, 2])
            x_max = np.minimum(bbox_1[:, 3].reshape(-1, 1), bbox_2[:, 3])
            y_max = np.minimum(bbox_1[:, 4].reshape(-1, 1), bbox_2[:, 4])
            z_max = np.minimum(bbox_1[:, 5].reshape(-1, 1), bbox_2[:, 5])
            intersection = (
                np.maximum(0, x_max - x_min)
                * np.maximum(0, y_max - y_min)
                * np.maximum(0, z_max - z_min)
            )
            # compute the union of the bounding boxes
            area1 = (
                (bbox_1[:, 3] - bbox_1[:, 0])
                * (bbox_1[:, 4] - bbox_1[:, 1])
                * (bbox_1[:, 5] - bbox_1[:, 2])
            )
            area2 = (
                (bbox_2[:, 3] - bbox_2[:, 0])
                * (bbox_2[:, 4] - bbox_2[:, 1])
                * (bbox_2[:, 5] - bbox_2[:, 2])
            )
            union = area1.reshape(-1, 1) + area2 - intersection
            # compute the IoU
            iou = intersection / union
            return iou

        iou_matrix = bbox_iou(gt_notion_bbox, ovi_notion_bbox)
        max_iou = np.max(iou_matrix, axis=0)
        max_row_indices = np.argmax(iou_matrix, axis=0)
        max_row_indices[max_iou < iou_thresh] = -1
        ovi2gt_map = {}
        gt2ovi_map = {}
        for i, max_row_index in enumerate(max_row_indices):
            if max_row_index != -1:
                if max_row_index in gt2ovi_map:
                    if max_iou[i] > gt2ovi_map[max_row_index][1]:
                        ovi2gt_map[i] = (max_row_index, max_iou[i])
                        gt2ovi_map[max_row_index] = (i, max_iou[i])
                else:
                    ovi2gt_map[i] = (max_row_index, max_iou[i])
                    gt2ovi_map[max_row_index] = (i, max_iou[i])

        # save the ovi2gt_map
        query_path = os.path.join(self.output_path(), "graph_queries")
        os.makedirs(query_path, exist_ok=True)
        gt_match = {"ovi2gt_map": ovi2gt_map, "gt2ovi_map": gt2ovi_map}

        with open(
            os.path.join(os.path.dirname(os.path.dirname(query_path)), "gt_match.pkl"), "wb"
        ) as f:
            pickle.dump(gt_match, f)

    def gen_query_data(self, use_gt_map: bool = True, enable_user: bool = False):
        """Generate data for evaluation"""
        print(f"Generating query data for {self.scene_map_cfg['scene_name']}...")
        self.build_from_ovi(use_gt_map=use_gt_map, enable_user=enable_user)
        query_path = os.path.join(self.output_path(), "graph_queries")
        os.makedirs(query_path, exist_ok=True)
        notion_count = self.notion_graph.notion_count
        ovi_gt = dict()
        user_relations = dict()
        # ovi_gt["users"] = []
        assert notion_count > 0
        for i in tqdm.tqdm(range(notion_count), desc=f'Generating Queries', unit='notion'):
            if self.notion_graph.notions[i].domain == Domain.USR:
                # ovi_gt["users"].append(i)
                user_name = self.notion_graph.notions[i].name
                user_relations[user_name] = dict()
                all_links = list(set(self.notion_graph.notions[i].links))
                for n2 in all_links:
                    if (i, n2) not in self.notion_graph.links:
                        link = self.notion_graph.links[(n2, i)]
                    else:
                        link = self.notion_graph.links[(i, n2)]
                    relation_descs = link.relation_desc
                    for rel in relation_descs:
                        if rel not in user_relations[user_name]:
                            user_relations[user_name][rel] = []
                        user_relations[user_name][rel].append(n2)
        ovi_gt = user_relations
        with open(os.path.join(query_path, "ovimap_gt.pkl"), "wb") as fp:
            pickle.dump(ovi_gt, fp)

        # generate queries
        queries = self.generate_query(qper_notion=3, qmax_len=3)
        # saving the pkl file containing the graph queries
        with open(os.path.join(query_path, "graph_queries.pkl"), "wb") as fp:
            pickle.dump(queries, fp)

    def gen_match_data(self, top_k: int, method: str = "prob", enable_user: bool = False):
        """Generate matching result
        Args:
            method: matching method, prob, kernel_thresh, kernel_gnn
        """
        # build map
        print(f"Generating match data for {self.scene_map_cfg['scene_name']}...")
        self.build_from_ovi(use_gt_map=False, enable_user=enable_user)
        query_path = os.path.join(self.output_path(), "graph_queries")
        query_file = os.path.join(query_path, "graph_queries.pkl") # graph_queries_50.pkl for 50 natural-language queries
        with open(query_file, "rb") as fp:
            data = pickle.load(fp)
        queries = data["queries"]
        address = data["address"]

        results = []
        tqdm_desc = f'Generating Match Data ({method})...'
        for i in tqdm.tqdm(range(len(queries)), desc=tqdm_desc, unit='matches'):
            query = queries[i]
            query_result = self.query(query_str=query, top_k=top_k, verbose=False, method=method)
            result = {}
            result["query"] = query
            result["candidates"] = [candidate.to_dict() for candidate in query_result["candidates"]]
            result["raw_candidates"] = [
                candidate.to_dict() for candidate in query_result["raw_candidates"]
            ]
            result["target"] = {
                "name": query_result["target_name"],
                "id": query_result["target_id"],
                "address": address[query_result["target_id"]],
            }
            results.append(result)

        with open(os.path.join(query_path, f"predictions_{method}.pkl"), "wb") as fp:
            pickle.dump(results, fp)
        with open(os.path.join(query_path, "predictions_base.pkl"), "wb") as fp:
            pickle.dump(results, fp)

    def gen_gnn_data(self, enable_user: bool = False):
        """Generate data for GNN training"""
        print(f"Generating GNN data for {self.scene_map_cfg['scene_name']}...")
        self.build_from_ovi(use_gt_map=False, enable_user=enable_user)
        query_path = os.path.join(self.output_path(), "graph_queries")
        root_data_path = os.path.dirname(self.scene_map_cfg["data_path"])
        gnn_path = os.path.join(self.output_path(), "gnn_matcher")
        # clean the directory
        os.makedirs(gnn_path, exist_ok=True)
        os.system("rm -rf " + gnn_path + "/*")
        # load the gt query and parse
        query_file_list = [f for f in os.listdir(query_path) if f.startswith("predictions_")]
        query_file = os.path.join(query_path, query_file_list[0])
        with open(query_file, "rb") as fp:
            predictions = pickle.load(fp)
        gt_file = os.path.join(os.path.dirname(os.path.dirname(query_path)), "gt_match.pkl")
        with open(gt_file, "rb") as fp:
            gt_match = pickle.load(fp)

        count = 0
        tqdm_desc = f'Generating GNN Data...'
        for i in tqdm.tqdm(range(len(predictions)), desc=tqdm_desc, unit='gnn_data'):
            prediction = predictions[i]
            query_str = prediction["query"]
            target_gt_id = prediction["target"]["id"]
            if target_gt_id not in gt_match["gt2ovi_map"]:
                continue
            else:
                target_ovi_id = gt_match["gt2ovi_map"][target_gt_id][0]
                # compute pos id & neg id
                candidate_ids = [candidate["id"] for candidate in prediction["candidates"]]
                raw_candidate_ids = [candidate["id"] for candidate in prediction["raw_candidates"]]
                if target_ovi_id not in candidate_ids:
                    continue  # no match
                else:
                    pos_id = target_ovi_id
                    neg_id_list = [
                        candidate_id for candidate_id in candidate_ids if candidate_id != pos_id
                    ]
            target_name, target_type, target_id = self.parse_syntax(self.notion_query, query_str)
            if self.notion_query.notion_count == 0 or target_type != "object":
                # FIXME: this is a bug in the query generation
                # FIXME: currently, only focusing on object queries
                continue
            neg_id = neg_id_list[0]  # use the closest negative sample
            # triplet
            (x_s, x_p, x_n, edge_index_s, edge_index_p, edge_index_n,) = self.notion_graph.generate(
                target="gnn_triplet",
                G_s=self.notion_query,
                id_p=pos_id,
                id_n=neg_id,
                use_padding=True,
            )
            torch.save(
                {
                    "x_s": x_s,
                    "x_p": x_p,
                    "x_n": x_n,
                    "edge_index_s": edge_index_s,
                    "edge_index_p": edge_index_p,
                    "edge_index_n": edge_index_n,
                },
                os.path.join(gnn_path, f"data_{count}.pt"),
            )
            count += 1

    def calc_3d_iou(self, gt_ind: List[int], pred_ind: List[int]):
        # gt_ind and pred_ind are two lists of indices
        # Calculate union and intersection and return iou
        gt_ind = set(gt_ind)
        pred_ind = set(pred_ind)
        union = gt_ind.union(pred_ind)
        intersection = gt_ind.intersection(pred_ind)
        iou = float(len(intersection)) / float(len(union))
        return round(iou, 3)

    def eval_query(self, use_gt_map: bool = False, use_relation: bool = True, method: str = "prob"):
        """Eval"""
        query_path = os.path.join(self.output_path(), "graph_queries")
        # load the query and parse
        query_file = os.path.join(query_path, "graph_queries.pkl") # graph_queries_50.pkl for 50 natural-language queries
        with open(query_file, "rb") as fp:
            data = pickle.load(fp)
        address = data["address"]
        # statistics
        top_1_iou_avg = 0
        top_3_iou_avg = 0

        query_file = os.path.join(query_path, f"predictions_{method}.pkl")
        gt_match_file = os.path.join(os.path.dirname(os.path.dirname(query_path)), "gt_match.pkl")
        with open(gt_match_file, "rb") as fp:
            gt_match = pickle.load(fp)

        with open(query_file, "rb") as fp:
            results = pickle.load(fp)
        num_queries = len(results)
        IoU_3d_arr = []
        mIoU_3d = []
        IoU_15 = 0
        IoU_25 = 0
        IoU_50 = 0
        IoU_75 = 0
        top1_matches = 0
        top3_matches = 0
        left_out_queries = 0
        upper_bound = 0
        class_wise_top1_matches = dict()
        class_wise_top3_matches = dict()
        class_wise_top1_misses = dict()
        class_wise_top3_misses = dict()

        iclass_wise_top1_matches = dict()
        iclass_wise_top3_matches = dict()
        iclass_wise_top1_misses = dict()
        iclass_wise_top3_misses = dict()

        with open(os.path.join(self.scene_map_cfg["data_path"], self.scene_map_cfg["scene_name"], "detic_output", self.scene_map_cfg["detic_exp"], "predictions", self.scene_map_cfg["annotation_file"]), "rb") as fp:
            pt_file = pickle.load(fp)
        ovir = pt_file.nodes(data=True)
        
        with open(os.path.join(self.scene_map_cfg["data_path"], self.scene_map_cfg["scene_name"], "detic_output", self.scene_map_cfg["detic_exp"], "predictions", self.scene_map_cfg["annotation_gt_file"]), "rb") as fp:
            gt_file = pickle.load(fp)

        for ind, result in enumerate(results):
            target_address = address[result["target"]["id"]]
            top_1_iou = 0
            top_3_iou = 0
            if method == "base":
                candidate_pool = result["raw_candidates"]
            else:
                candidate_pool = result["candidates"]
            if len(candidate_pool) > 2:
                iou3d = self.calc_3d_iou(gt_file[int(result["target"]["id"])]["pt_indices"], ovir[int(candidate_pool[0]["id"])]["pt_indices"])
                IoU_3d_arr.append(iou3d)
                if iou3d > 0.15:
                    IoU_15 += 1
                if iou3d > 0.25:
                    IoU_25 += 1
                if iou3d > 0.5:
                    IoU_50 += 1
                if iou3d > 0.75:
                    IoU_75 += 1
                pass    
                
                if result["target"]["id"] in gt_match["gt2ovi_map"]:
                    upper_bound += 1
                found_top3_match = False
                found_top1_match = False
                for j in range(3):
                    iou_target, iou_queried = p9.iou(target_address, candidate_pool[j]["address"])
                    # update iou
                    iou_j = min(iou_target, iou_queried)
                    if j == 0:
                        if candidate_pool[j]["id"] in gt_match["ovi2gt_map"]:
                            if (
                                gt_match["ovi2gt_map"][candidate_pool[j]["id"]][0]
                                == result["target"]["id"]
                            ):
                                top1_matches += 1
                                top3_matches += 1
                                if result["target"]["name"] not in class_wise_top1_matches:
                                    class_wise_top1_matches[result["target"]["name"]] = 1
                                else:
                                    class_wise_top1_matches[result["target"]["name"]] += 1

                                if result["target"]["name"] not in class_wise_top3_matches:
                                    class_wise_top3_matches[result["target"]["name"]] = 1
                                else:
                                    class_wise_top3_matches[result["target"]["name"]] += 1

                                if result["target"]["name"] not in iclass_wise_top1_matches:
                                    iclass_wise_top1_matches[result["target"]["name"]] = [ind]
                                else:
                                    iclass_wise_top1_matches[result["target"]["name"]].append(ind)

                                if result["target"]["name"] not in iclass_wise_top3_matches:
                                    iclass_wise_top3_matches[result["target"]["name"]] = [ind]
                                else:
                                    iclass_wise_top3_matches[result["target"]["name"]].append(ind)

                                found_top3_match = True
                                found_top1_match = True
                        top_1_iou = iou_j if top_1_iou < iou_j else top_1_iou
                    if j < 3:
                        if (
                            not found_top3_match
                            and candidate_pool[j]["id"] in gt_match["ovi2gt_map"]
                        ):
                            if (
                                gt_match["ovi2gt_map"][candidate_pool[j]["id"]][0]
                                == result["target"]["id"]
                            ):
                                top3_matches += 1
                                found_top3_match = True
                                if result["target"]["name"] not in class_wise_top3_matches:
                                    class_wise_top3_matches[result["target"]["name"]] = 1
                                else:
                                    class_wise_top3_matches[result["target"]["name"]] += 1

                                if result["target"]["name"] not in iclass_wise_top3_matches:
                                    iclass_wise_top3_matches[result["target"]["name"]] = [ind]
                                else:
                                    iclass_wise_top3_matches[result["target"]["name"]].append(ind)

                        top_3_iou = iou_j if top_3_iou < iou_j else top_3_iou
                # finally, we just compare the result
                if not found_top1_match:
                    if result["target"]["name"] not in class_wise_top1_misses:
                        class_wise_top1_misses[result["target"]["name"]] = 1
                    else:
                        class_wise_top1_misses[result["target"]["name"]] += 1

                    if result["target"]["name"] not in iclass_wise_top1_misses:
                        iclass_wise_top1_misses[result["target"]["name"]] = [ind]
                    else:
                        iclass_wise_top1_misses[result["target"]["name"]].append(ind)

                if not found_top3_match:
                    if result["target"]["name"] not in class_wise_top3_misses:
                        class_wise_top3_misses[result["target"]["name"]] = 1
                    else:
                        class_wise_top3_misses[result["target"]["name"]] += 1

                    if result["target"]["name"] not in iclass_wise_top3_misses:
                        iclass_wise_top3_misses[result["target"]["name"]] = [ind]
                    else:
                        iclass_wise_top3_misses[result["target"]["name"]].append(ind)
            else:
                left_out_queries += 1

            # update iou_avg
            top_1_iou_avg += top_1_iou
            top_3_iou_avg += top_3_iou
        # top_1_iou_avg /= num_queries - left_out_queries
        # top_3_iou_avg /= num_queries - left_out_queries
        print(f"top_1_iou_avg: {top_1_iou_avg}")
        print(f"top_3_iou_avg: {top_3_iou_avg}")
        mIoU_3d = round( float(sum(IoU_3d_arr))  / len(IoU_3d_arr), 3)
        print(f"mIoU_3d: {mIoU_3d}")
        IoU_15_val = IoU_15
        print(f"IoU_15_val: {IoU_15_val}")
        IoU_25_val = IoU_25
        print(f"IoU_25_val: {IoU_25_val}")
        IoU_50_val = IoU_50
        print(f"IoU_50_val: {IoU_50_val}")
        IoU_75_val = IoU_75
        print(f"IoU_75_val: {IoU_75_val}")
        IoU_15 = round( (IoU_15*100) / len(IoU_3d_arr), 2)
        print(f"IoU_15: {IoU_15}%")
        IoU_25 = round( (IoU_25*100) / len(IoU_3d_arr), 2)
        print(f"IoU_25: {IoU_25}%")
        IoU_50 = round( (IoU_50*100) / len(IoU_3d_arr), 2)
        print(f"IoU_50: {IoU_50}%")
        IoU_75 = round( (IoU_75*100) / len(IoU_3d_arr), 2)
        print(f"IoU_75: {IoU_75}%")
        # print(sorted(IoU_3d_arr, reverse=True))
        res = {
            "mIoU_3d": mIoU_3d,
            "IoU_15": IoU_15,
            "IoU_25": IoU_25,
            "IoU_50": IoU_50,
            "IoU_75": IoU_75,
            "IoU_3d_arr": IoU_3d_arr,
            "IoU_15_val": IoU_15_val,
            "IoU_25_val": IoU_25_val,
            "IoU_50_val": IoU_50_val,
            "IoU_75_val": IoU_75_val,
            "top_1_iou_avg": top_1_iou_avg,
            "top_3_iou_avg": top_3_iou_avg,
            "top1_matches": top1_matches,
            "top3_matches": top3_matches,
            "num_queries": num_queries - left_out_queries,
            "top1_match_rate": top1_matches * 100 // (num_queries - left_out_queries),
            "top3_match_rate": top3_matches * 100 // (num_queries - left_out_queries),
            "upper_bound": (upper_bound*100)//(num_queries - left_out_queries),
            "class_wise_top1_matches": class_wise_top1_matches,
            "class_wise_top3_matches": class_wise_top3_matches,
            "class_wise_top1_misses": class_wise_top1_misses,
            "class_wise_top3_misses": class_wise_top3_misses,
            "iclass_wise_top1_matches": iclass_wise_top1_matches,
            "iclass_wise_top3_matches": iclass_wise_top3_matches,
            "iclass_wise_top1_misses": iclass_wise_top1_misses,
            "iclass_wise_top3_misses": iclass_wise_top3_misses
        }
        assert res["upper_bound"] >= res["top1_match_rate"]
        assert res["upper_bound"] >= res["top3_match_rate"]
        # print('top3_match_rate:', res['top3_match_rate'], 'top1_match_rate:', res['top1_match_rate'] )
        with open(os.path.join(query_path, f"results_{method}.pkl"), "wb") as fp:
            pickle.dump(res, fp)
        print(f"\n************** {method} **************\n")
        print(
            f"top1_match_rate: {res['top1_match_rate']}, top1_matches: {top1_matches} out of {num_queries - left_out_queries} queries...."
        )
        print(
            f"top3_match_rate: {res['top3_match_rate']}, top3_matches: {top3_matches} out of {num_queries - left_out_queries} queries..."
        )
        print("\n******************************\n")

    def eval(self, task_name, **kwargs):
        """eval task, task_name will be informat of 'task_type:task_file'"""
        task_type, task_scene = task_name.split(":")
        if task_type == "query":
            self.eval_query()
        elif task_type == "gen_query":
            self.scene_map_cfg["scene_name"] = task_scene

            self.eval_enable_user = True
            self.gen_gt_match(iou_thresh=self.eval_iou_thresh)

            self.gen_query_data(use_gt_map=True, enable_user=self.eval_enable_user)
            self.gen_match_data(
                top_k=self.eval_top_k, method="jaccard", enable_user=self.eval_enable_user
            )
            self.gen_match_data(
                top_k=self.eval_top_k,
                method="szymkiewicz_simpson",
                enable_user=self.eval_enable_user,
            )
            self.gen_match_data(
                top_k=self.eval_top_k, method="prob", enable_user=self.eval_enable_user
            )

            self.eval_query(method="jaccard")
            self.eval_query(method="szymkiewicz_simpson")
            self.eval_query(method="prob")
            self.eval_query(method="base")

            self.eval_enable_user = False
            
            self.gen_query_data(use_gt_map=True, enable_user=self.eval_enable_user)
            self.gen_match_data(
                top_k=self.eval_top_k, method="jaccard", enable_user=self.eval_enable_user
            )
            self.gen_match_data(
                top_k=self.eval_top_k,
                method="szymkiewicz_simpson",
                enable_user=self.eval_enable_user,
            )
            self.gen_match_data(
                top_k=self.eval_top_k, method="prob", enable_user=self.eval_enable_user
            )
            self.eval_query(method="jaccard")
            self.eval_query(method="szymkiewicz_simpson")
            self.eval_query(method="prob")
            self.eval_query(method="base")
        return "", True
