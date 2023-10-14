from __future__ import annotations
import os
import numpy as np
from PIL import Image
from enum import Enum
import open3d as o3d
import hydra
from ovsg.env.algo.notion import (
    Notion,
    NotionGraph,
    NotionLink,
    NotionGraphWrapper,
    Space,
    Feature,
    FeatureType,
)
from ovsg.utils.spatial.spatial import Points9 as p9
from ovsg.utils.spatial.spatial_encoder import SpatialRelEncoder, SpatialRelModel


class SpatialRelation(Enum):
    """Spatial Relationship"""

    NONE = 0
    NEAR = 1
    IN = 2
    ON = 3
    BELOW = 4


def compute_relationships(positions, near_thres, overlap_thres):
    n = positions.shape[0]
    relationships = np.zeros((n, n), dtype=np.int32)

    # Compute distances between all pairs of nodes
    dist = np.linalg.norm(positions[:, np.newaxis] - positions, axis=-1)
    dist_xy = np.linalg.norm(positions[:, np.newaxis, :2] - positions[:, :2], axis=-1)

    # Set relationships based on thresholds
    relationships[dist < near_thres] = SpatialRelation.NEAR.value
    return relationships


class NotionGraphSpatialWrapper(NotionGraphWrapper):
    """Wrapper for Notions, providing parse method for spatial address"""

    def __init__(
        self,
        notion_graph: NotionGraph | NotionGraphWrapper,
        near_thres: float,
        overlap_thres: float,
    ):
        self.notion_graph = notion_graph
        self.notion_pos = []
        # thresholds
        self.near_thres = near_thres
        self.overlap_thres = overlap_thres
        # create spatial encoder
        root_abs_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
        spe_cfg = hydra.compose(config_name="ml/spatial_encoder")
        spatial_rel_model = SpatialRelModel(
            spatial_input_size=spe_cfg.spatial_input_size,
            txt_input_size=spe_cfg.txt_input_size,
            hidden_size=spe_cfg.hidden_size,
            embedding_size=spe_cfg.embedding_size,
        ).to(spe_cfg.device)
        self.spatial_encoder = SpatialRelEncoder(
            spe_cfg,
            root_abs_path,
            spatial_rel_model,
        )
        self.spatial_encoder.load()

    def reset(self):
        """Reset NotionGraphWrapper"""
        self.notion_pos = []
        self.notion_graph.reset()

    def add_notion(
        self,
        keys: list[Feature],
        notion: any,
        address: str,
        domain: str,
        name: str,
        space: Space = Space.VIRTUAL,
        **kwargs
    ) -> int:
        """Add notion to Notions"""
        id = super().add_notion(keys, notion, address, domain, name, space, **kwargs)
        self.notion_pos.append(address)
        # check
        assert id == len(self.notion_pos) - 1, "Page ID not match"
        return id

    def update(self, **kwargs):
        """Update links between notions"""
        # call build_links of Notions
        super().update()
        # build spatial graph
        self._build_spatial_graph()

    def encode_spatial(self, pose: np.ndarray, **kwargs):
        """Encode spatial rel into a vector"""
        spatial_feature = self.spatial_encoder.encode_spatial(pose).detach().cpu().numpy()
        return Feature(feature=spatial_feature, feature_type=FeatureType.SPATIAL)

    def spatial_sprob(self, feature_1: Feature, feature_2: Feature):
        """Compute spatial similarity"""
        if (feature_1.feature_type, feature_2.feature_type) == (
            FeatureType.SPATIAL,
            FeatureType.CLIPTXT,
        ):
            spatial_embedding = feature_1.feature
            txt_embedding = feature_2.feature
        elif (feature_1.feature_type, feature_2.feature_type) == (
            FeatureType.CLIPTXT,
            FeatureType.SPATIAL,
        ):
            spatial_embedding = feature_2.feature
            txt_embedding = feature_1.feature
        else:
            print(feature_1.feature_type, feature_2.feature_type)
            print(Warning("Invalid feature type for spatial sprob"))
            return 0.0
        # reshape
        spatial_embedding = spatial_embedding.reshape(1, -1)
        txt_embedding = txt_embedding.reshape(1, -1)
        return (
            self.spatial_encoder.predict_from_embedding(spatial_embedding, txt_embedding)
            .detach()
            .cpu()
            .numpy()[0, 0]
        )

    # Main interface
    def generate(self, target: str, **kwargs):
        """Main interface"""
        if target == "spatial_pose":
            assert "pose_pair" in kwargs and "txt_embedding_dict" in kwargs
            pose_pair = kwargs["pose_pair"]
            txt_embedding_dict = kwargs["txt_embedding_dict"]
            return self.spatial_encoder.predict(
                pose_pair=pose_pair, txt_embedding_dict=txt_embedding_dict
            )
        elif target == "spatial_sprob":
            assert "feature_1" in kwargs and "feature_2" in kwargs
            feature_1 = kwargs["feature_1"]
            feature_2 = kwargs["feature_2"]
            return self.spatial_sprob(feature_1, feature_2)
        elif target == "notion_pos":
            return self.notion_pos
        else:
            # pass to other wrapper
            return self.notion_graph.generate(target, **kwargs)

    # Utils
    def visualize(self, **kwargs):
        """Visualize the spatial graph"""
        use_3d = kwargs.get("use_3d", False)
        if use_3d:
            background = kwargs.get("background", None)
            # visualize spatial graph using open3d
            # prepre data
            node_pose = []
            static_nodes = []
            links = []
            for i in range(len(self.notion_pos)):
                node_pose.append(self.notion_pos[i])
                if self.notions[i].space == Space.STATIC:
                    static_nodes.append(i)
                    if self.notion_pos[i] == [0, 0, 0]:
                        print("Warning: static notion has zero position")
                        print(self.notions[i].name)
            for (i, j) in self.links.keys():
                if i in static_nodes and j in static_nodes:
                    # only add links between static notions
                    links.append((i, j))

            if len(links) == 0:
                print("No links found")
                return

            pcd = o3d.geometry.PointCloud()
            pcd.points = o3d.utility.Vector3dVector(np.array(node_pose))
            pcd.paint_uniform_color([0, 1, 0])

            line_set = o3d.geometry.LineSet()
            line_set.points = pcd.points
            line_set.lines = o3d.utility.Vector2iVector(np.array(links))

            # set the colors for the lines (optional)
            colors = [[1, 0, 0] for _ in range(len(links))]
            line_set.colors = o3d.utility.Vector3dVector(np.array(colors))

            # visualize the spatial graph
            if background is not None:
                o3d.visualization.draw_geometries([background, pcd, line_set])
            else:
                o3d.visualization.draw_geometries([pcd, line_set])
        else:
            super().visualize(**kwargs)

    def _build_spatial_graph(self):
        """Build spatial graph based on notion positions"""
        # build spatial graph
        notion_pos_np = np.vstack(self.notion_pos)
        # compute distance
        relationships = compute_relationships(
            notion_pos_np[:, :3], self.near_thres, self.overlap_thres
        )
        # add links between static & dynamic notions
        for i in range(relationships.shape[0]):
            for j in range(relationships.shape[1]):
                if i > j:
                    # if i, j are both static/dynamic notion
                    if (self.notion_graph.notions[i].space in [Space.STATIC, Space.DYNAMIC]) and (
                        self.notion_graph.notions[j].space in [Space.STATIC, Space.DYNAMIC]
                    ):
                        notion_i = self.notion_graph.notions[i]
                        notion_j = self.notion_graph.notions[j]
                        if relationships[i, j] == SpatialRelation.NEAR.value:
                            if self.spatial_encoder is None:
                                # if there are spatial encoder, use it to check the link
                                self._link_node_manually(notion_i, notion_j)
                            else:
                                # ij
                                pose_ij = p9.normalize_pair(notion_i.address, notion_j.address)
                                spatial_key = self.encode_spatial(pose=pose_ij)
                                self.notion_graph.link(
                                    notion_i, "spatial", notion_j, relation_keys=spatial_key
                                )
                                # ji
                                pose_ji = p9.normalize_pair(notion_j.address, notion_i.address)
                                spatial_key = self.encode_spatial(pose=pose_ji)
                                self.notion_graph.link(
                                    notion_j, "spatial", notion_i, relation_keys=spatial_key
                                )
                        else:
                            continue
        # add links between (static / dynamic) and region notions
        for notion in self.notion_graph.notions:
            if notion.space == Space.STATIC or notion.space == Space.DYNAMIC:
                for region_notion in self.notion_graph.at_space(Space.REGION):
                    # in 3d
                    if p9.is_in(notion.address, region_notion.address):
                        if self.spatial_encoder is None:
                            self.notion_graph.link(notion, "in", region_notion)
                        else:
                            # notion->region
                            pose_pair = p9.normalize_pair(notion.address, region_notion.address)
                            spatial_key = self.encode_spatial(pose=pose_pair)
                            self.notion_graph.link(
                                notion, "spatial", region_notion, relation_keys=spatial_key
                            )

    def _link_node_manually(self, notion_i, notion_j):
        # near is symmetric
        self.notion_graph.link(notion_i, "near", notion_j)
        self.notion_graph.link(notion_j, "near", notion_i)
        # Manually compute relationship
        # if i is near to j, compute other relations
        view_angle = np.pi / 2.0
        # view_angle = 0.0
        # 1. on
        if p9.is_on(notion_i.address, notion_j.address):
            self.notion_graph.link(notion_i, "on", notion_j)
        elif p9.is_on(notion_j.address, notion_i.address):
            self.notion_graph.link(notion_j, "on", notion_i)

        # 2. above
        if p9.is_above(notion_i.address, notion_j.address):
            self.notion_graph.link(notion_i, "above", notion_j)
        elif p9.is_above(notion_j.address, notion_i.address):
            self.notion_graph.link(notion_j, "above", notion_i)

        # 3. under
        if p9.is_under(notion_i.address, notion_j.address):
            self.notion_graph.link(notion_i, "under", notion_j)
        elif p9.is_under(notion_j.address, notion_i.address):
            self.notion_graph.link(notion_j, "under", notion_i)

        # 4. left
        if p9.is_left(notion_i.address, notion_j.address, view_angle):
            self.notion_graph.link(notion_i, "left", notion_j)
        elif p9.is_left(notion_j.address, notion_i.address, view_angle):
            self.notion_graph.link(notion_j, "left", notion_i)

        # 5. right
        if p9.is_right(notion_i.address, notion_j.address, view_angle):
            self.notion_graph.link(notion_i, "right", notion_j)
        elif p9.is_right(notion_j.address, notion_i.address, view_angle):
            self.notion_graph.link(notion_j, "right", notion_i)

        # 6. front
        if p9.is_front(notion_i.address, notion_j.address, view_angle):
            self.notion_graph.link(notion_i, "front", notion_j)
        elif p9.is_front(notion_j.address, notion_i.address, view_angle):
            self.notion_graph.link(notion_j, "front", notion_i)

        # 7. behind
        if p9.is_behind(notion_i.address, notion_j.address, view_angle):
            self.notion_graph.link(notion_i, "behind", notion_j)
        elif p9.is_behind(notion_j.address, notion_i.address, view_angle):
            self.notion_graph.link(notion_j, "behind", notion_i)
