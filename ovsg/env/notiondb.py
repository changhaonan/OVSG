from __future__ import annotations
import traceback
from PIL import Image
import cv2
import glob
import numpy as np
from ovsg.core.env import EnvBase
import open3d as o3d
import copy
import tqdm
import os
import random
import re
from ovsg.env.algo.notion_encoder import NotionGraphEncoder
from ovsg.env.algo.notion import (
    NotionGraph,
    NotionGraphWrapper,
    Space,
    Notion,
    User,
    Domain,
    Feature,
    FeatureType,
)
from ovsg.utils.spatial.spatial import Points9 as p9
from ovsg.utils.spatial.spatial_encoder import build_txt_embedding
from ovsg.env.algo.notion_spatial import NotionGraphSpatialWrapper
from ovsg.env.algo.notion_kernel import NotionKernelWrapper
from ovsg.env.ovimap.class_labels_utils import class_labels_n_ids


class NotionDB(EnvBase):
    """Notion Data Base"""

    def __init__(self, cfg):
        super().__init__(cfg)
        self.notion_memory = dict()
        self.txt_embedding_dict = build_txt_embedding("clip", "cuda")
        # init notion
        if cfg.notion_base == "default":
            self.notion_graph = NotionGraphEncoder(cfg)
            self.notion_query = NotionGraphEncoder(cfg)
        else:
            raise NotImplementedError
        # wrapper
        if cfg.notion_address == "spatial":
            self.notion_graph = NotionGraphSpatialWrapper(
                self.notion_graph, cfg.notion_near_thres, cfg.notion_overlap_thres
            )
        if cfg.gnn_padding > 0:
            self.notion_graph = NotionKernelWrapper(
                self.notion_graph, cfg.gnn_padding, cfg.gnn_max_node_size
            )

        # init for test
        self.debug_flag = True if cfg.debug else False
        self.notion_dir = cfg.notion_dir
        # user related
        self.user_default_photo = cv2.imread(cfg.user_default_photo)
        # params
        self.rebuild_region = cfg.rebuild_region

    def reset(self):
        """Reset NotionDB"""
        self.notion_memory = dict()
        self.notion_graph.reset()

    # extended notion method
    def add_notion(
        self,
        keys: list[Feature],
        content: any,
        pos: np.ndarray,
        domain: Domain,
        name: str,
        image=None,
        space: Space = Space.VIRTUAL,
        tags=list[str] | None,
        **kwargs,
    ) -> int:
        """Add instance to NotionGraph
        Args:
            keys: list of keys
            content: content of the instance
            pos: position of the instance
            domain: domain of the instance
            name: name of the instance
            image: image of the instance
            space: space of the instance
            tags: tags of the instance
            **kwargs: other arguments
        Returns:
            notion id
        """
        # add instance notion
        inst_id = self.notion_graph.add_notion(
            keys=keys,
            notion=content,
            address=pos,
            domain=domain,
            name=name,
            space=space,
            **kwargs,
        )
        return inst_id

    def add_user(
        self,
        pos: np.ndarray,
        names: list[str] | str,
        image: np.ndarray | Image.Image | None = None,
        **kwargs,
    ):
        """Add user to NotionGraph
        Args:
            names: list of names(nick name), or a single name,
            image: image of the user
            **kwargs: other arguments
        """
        if isinstance(names, str):
            names = [names]
        if image is None:
            image = self.user_default_photo
        user = User(names=names, photo=image)
        self.add_notion(
            keys=[],
            content=user,
            pos=pos,
            domain=Domain.USR,
            name=names[0],
            image=None,
            space=Space.DYNAMIC,
            **kwargs,
        )

    def parse_syntax(self, notion_graph: NotionGraph | NotionGraphWrapper, query_syntax: str):
        """Build links from query syntax"""
        notion_graph.reset()
        lines = query_syntax.strip().split("\n")
        notion_dict = {}
        target_name = None
        target_type = None
        target_id = None
        # parse notion
        for line in lines:
            if "@" in line:
                prefix, notion_str = line.split("@")
                prefix = prefix.strip()
                notion_str = notion_str.split("{")[0].strip()
                type_str = line.split("{")[1].split("}")[0].strip().split("#")[0].strip()
                if len(line.split("{")[1].split("}")[0].strip().split("#")) > 1:
                    id_str = line.split("{")[1].split("}")[0].strip().split("#")[1].strip()
                else:
                    id_str = None
                if prefix == "target":
                    target_name = notion_str
                    target_type = type_str
                    target_id = int(id_str) if id_str is not None else None
                    if target_name not in notion_dict:
                        if target_type == "user":
                            domain = Domain.USR
                        elif target_type == "object":
                            domain = Domain.INS
                        elif target_type == "region":
                            domain = Domain.RGN
                        elif target_type == "tag":
                            domain = Domain.TAG
                        else:
                            raise ValueError(f"Unknown notion type: {target_type}")
                        notion_dict[target_name] = notion_graph.add_notion(
                            keys=[],
                            notion=target_name,
                            address=p9.zeros(),
                            name=target_name,
                            domain=domain,
                            space=Space.VIRTUAL,
                        )

        # parse relation
        for line in lines:
            if "--" in line:
                notion1, relation, notion2 = line.split("--")
                notion_names = [notion1.split("{")[0].strip(), notion2.split("{")[0].strip()]
                notion_types = [
                    notion1.split("{")[1].split("}")[0].strip().split("#")[0].strip(),
                    notion2.split("{")[1].split("}")[0].strip().split("#")[0].strip(),
                ]

                # check if notion1 and notion2 are defined
                for notion_name, notion_type in zip(notion_names, notion_types):
                    if notion_name not in notion_dict:
                        if notion_type == "user":
                            domain = Domain.USR
                        elif notion_type == "object":
                            domain = Domain.INS
                        elif notion_type == "region":
                            domain = Domain.RGN
                        elif notion_type == "tag":
                            domain = Domain.TAG
                        else:
                            raise ValueError(f"Unknown notion type: {notion_type}")
                        notion_dict[notion_name] = notion_graph.add_notion(
                            keys=[],
                            notion=notion_name,
                            address=p9.zeros(),
                            name=notion_name,
                            domain=domain,
                            space=Space.VIRTUAL,
                        )

                relation = relation.strip()
                # further parse relation
                pattern = r"^\s*(\w.*\w)\s*\[\s*(\w.*\w)\s*\]\s*$"
                match = re.match(pattern, relation)
                relation_type = match.group(2)
                notion2 = notion2.strip()
                if relation_type == "spatial":
                    relation_key = self.notion_graph.encode_text(
                        text=f"A is {match.group(1)} B", encoder="clip"
                    )
                    notion_graph.link(
                        notion1=notion_graph.notions[notion_dict[notion_names[0]]],
                        relation="spatial",
                        notion2=notion_graph.notions[notion_dict[notion_names[1]]],
                        confidence=1.0,
                        relation_keys=relation_key,
                    )
                else:
                    notion_graph.link(
                        notion1=notion_graph.notions[notion_dict[notion_names[0]]],
                        relation=match.group(1),
                        notion2=notion_graph.notions[notion_dict[notion_names[1]]],
                        confidence=1.0,
                    )
        return target_name, target_type, target_id

    # matching algorithm
    def graph_kernel_prob(
        self,
        notion_graph: NotionGraph | NotionGraphWrapper,
        notion_query:  NotionGraph | NotionGraphWrapper,
        target: str,
        top_k: int,
        domain: Domain | None = None,
        space: Space | None = None,
        verbose: bool = False,
        **kwargs,
    ):
        """Naive matching algorithm
        Args:
            notion_graph: notion graph
            notion_query: query notion graph
            target: target notion name
            top_k: number of results to return
            domain: domain of the target notion
            space: space of the target notion
            verbose: verbose
        Returns:
            sorted_sprobs: sorted similarity probabilities
            sorted_candidates: sorted candidates by sprob
            sorted_matches: sorted matches by sprob
            candidates: raw candidates before sorting
        """

        def compute_link_sprob(link1, link2, notion1, notion2):
            # compute similarity between two links
            # get notion_graph
            notion11 = notion1.notions[link1.id1]
            notion12 = notion1.notions[link1.id2]
            notion21 = notion2.notions[link2.id1]
            notion22 = notion2.notions[link2.id2]

            # The order matters
            sprob_match = notion11.sprob(notion21) * notion12.sprob(notion22)
            # sprob_match = 1.0  # set to 1.0 to test the relation match
            # relation match
            sprob_relation = 0.0
            for relation_desc1, relation_key1 in zip(link1.relation_desc, link1.relation_key):
                for relation_desc2, relation_key2 in zip(link2.relation_desc, link2.relation_key):
                    sprob_relation = max(sprob_relation, relation_key1.sprob(relation_key2))
                    # deal with spatial relation sperately
                    if relation_desc1 == "spatial" and relation_desc2 == "spatial":
                        sprob_relation = self.notion_graph.generate(
                            target="spatial_sprob", feature_1=relation_key1, feature_2=relation_key2
                        )
            # print(f"sprob_relation: {sprob_relation}")
            sprob_relation = max(0.0, sprob_relation)  # relation similarity should be positive

            # debug
            if verbose:
                print("------------------")
                link1.view()
                link2.view()
                print(f"sprob: {sprob_match * sprob_relation}")
                print("==================")
            return sprob_match * sprob_relation

        query_links = notion_query.links
        # get candidate
        candidates, sprobs = notion_graph.search(
            query=target, top_k=top_k, domain=domain, space=space, **kwargs
        )
        match_sprobs = []
        matches = []
        for candidate, sprob in zip(candidates, sprobs):
            _, links = self.notion_graph.subgraph_nodelink(candidate)
            # compute sprobilarity with query notion_graph
            # sprob can only be computed at the same domain
            # compare two links
            match_sprob = 1.0
            candi_match = {}
            for query_link in query_links.values():
                max_link_sprob = -np.inf
                for link in links:
                    # compute sprobilarity between two links
                    link_sprob = compute_link_sprob(query_link, link, notion_query, notion_graph)
                    if link_sprob > max_link_sprob:
                        max_link_sprob = link_sprob
                        candi_match[query_link] = link
                # each query link should have at least prob of 0.1
                match_sprob *= max(max_link_sprob, 0.1)
                if verbose:
                    print("********************")
            match_sprobs.append(match_sprob)
            matches.append(candi_match)
            if verbose:
                print(f"candidate: {candidate.name}, sprob: {match_sprob}")
        # sorted candidate
        sorted_sprobs = sorted(match_sprobs, reverse=True)
        sorted_candidates = [
            candidate
            for _, candidate in sorted(
                zip(match_sprobs, candidates), key=lambda x: x[0], reverse=True
            )
        ]
        sorted_matches = [
            candi_match
            for _, candi_match in sorted(
                zip(match_sprobs, matches), key=lambda x: x[0], reverse=True
            )
        ]
        return sorted_sprobs, sorted_candidates, sorted_matches, candidates

    def graph_kernel(
        self,
        notion_graph:  NotionGraph | NotionGraphWrapper,
        notion_query: NotionGraph | NotionGraphWrapper,
        target: str,
        top_k: int,
        domain:  Domain | None = None,
        space:  Space | None = None,
        kernel_method: str = "jaccard",
        verbose: bool = False,
        **kwargs,
    ):
        """Graph kernel method: gnn and prob are implemented separately"""
        # get candidate
        candidates, sprobs = notion_graph.search(
            query=target, top_k=top_k, domain=domain, space=space, **kwargs
        )
        candidate_ids = [candidate.id for candidate in candidates]
        # rank by gnn
        dists = self.notion_graph.generate(
            "kernel_dist", G_s=notion_query, candid_ids=candidate_ids, kernel_method=kernel_method
        )
        sorted_dists = sorted(dists)
        sorted_candidates = [
            candidate for _, candidate in sorted(zip(dists, candidates), key=lambda x: x[0])
        ]
        # debug
        sorted_candidate_ids = [candidate.id for candidate in sorted_candidates]
        return sorted_dists, sorted_candidates, [], candidates

    # main interface
    def chat(
        self,
        str_msg: str,
        img_msg: list[Image.Image] | list[np.ndarray] | None = None,
        **kwargs,
    ) -> tuple[str, bool]:
        self.parse_syntax(self.notion_query, str_msg)
        self.notion_query.visualize()
        return "Success", True

    def get_notion_type(self, domain):
        """Get notion type"""
        if domain == Domain.USR:
            return "user"
        elif domain == Domain.INS:
            return "object"
        elif domain == Domain.RGN:
            return "region"
        elif domain == Domain.TAG:
            return "tag"

    def generate_graph_query(self, n1, n2):
        """Generate graph query for node n1 and n2"""

        def parse_pose(pose, rdesc):
            # random.shuffle(pose)
            for i in range(10):
                if pose[i][1] > 0.9:
                    for pose_txt in pose[i][0]:
                        return pose_txt[4:-2].strip() + " [" + rdesc + "] -- "
            return None

        query_str = ""
        if (n1, n2) not in self.notion_graph.links:
            n1, n2 = n2, n1
        link = self.notion_graph.links[(n1, n2)]
        relation_desc = random.choice(link.relation_desc)
        if (n1, n2, relation_desc) not in self.notion_memory:
            notion_type_1 = self.get_notion_type(self.notion_graph.notions[n1].domain)
            notion_type_2 = self.get_notion_type(self.notion_graph.notions[n2].domain)
            if relation_desc == "spatial":
                # find its nearest spatial relation
                pose_pair = p9.normalize_pair(
                    self.notion_graph.notions[n1].address,
                    self.notion_graph.notions[n2].address,
                )
                pose_txt_list = self.notion_graph.generate(
                    target="spatial_pose",
                    pose_pair=pose_pair,
                    txt_embedding_dict=self.txt_embedding_dict,
                )
                spatial_relation = parse_pose(pose_txt_list, relation_desc)
                if spatial_relation is not None:
                    query_str += (
                        link.notion_name1 + " {" + notion_type_1 + f" #{link.id1}" + "}" + " -- "
                    )
                    query_str += spatial_relation
                    query_str += (
                        f"{link.notion_name2} " + "{" + notion_type_2 + f" #{link.id2}" + "}" + "\n"
                    )
                    self.notion_memory[(n1, n2, relation_desc)] = query_str
            elif relation_desc == "tag":
                query_str += (
                    link.notion_name1 + " {" + notion_type_1 + f" #{link.id1}" + "}" + " -- "
                )
                query_str += "tag [tag] -- "
                query_str += (
                    f"{link.notion_name2} " + "{" + notion_type_2 + f" #{link.id2}" + "}" + "\n"
                )
                self.notion_memory[(n1, n2, relation_desc)] = query_str
            else:
                query_str += (
                    link.notion_name1 + " {" + notion_type_1 + f" #{link.id1}" + "}" + " -- "
                )
                query_str += f"{relation_desc} [{relation_desc}] -- "
                query_str += (
                    f"{link.notion_name2} " + "{" + notion_type_2 + f" #{link.id2}" + "}" + "\n"
                )
                self.notion_memory[(n1, n2, relation_desc)] = query_str
        else:
            query_str += self.notion_memory[(n1, n2, relation_desc)]
        return query_str

    # eval method
    def generate_query(
        self, qper_notion=10, qmax_len=4, skip_regions=True, skip_users=True
    ) -> list[str]:
        """Generate query"""
        # get spatial relationships
        graph_queries = dict()
        graph_queries["queries"] = []
        graph_queries["address"] = []
        notion_count = self.notion_graph.notion_count
        assert notion_count > 0
        for i in tqdm.tqdm(range(notion_count)):
            graph_queries["address"].append(self.notion_graph.notions[i].address)
            # if skip_regions and self.notion_graph.notions[i].name in class_labels_n_ids["scannet"]["BLACK_LIST"]:
            #     continue
            # elif skip_users and self.get_notion_type(self.notion_graph.notions[i].domain) == "user":
            #     continue
            # elif self.get_notion_type(self.notion_graph.notions[i].domain) == "region":
            #     continue
            if self.get_notion_type(self.notion_graph.notions[i].domain) == "object":
                # if self.get_notion_type(self.notion_graph.notions[i].domain) == "object" and self.notion_graph.notions[i].name not in class_labels_n_ids["scannet"]["BLACK_LIST"]:
                # if True:
                all_links = list(set(self.notion_graph.notions[i].links))
                query_head = (
                    f"target @ {self.notion_graph.notions[i].name}"
                    + " {"
                    + self.get_notion_type(self.notion_graph.notions[i].domain)
                    + f" #{i}"
                    + "}"
                    + "\n"
                )
                # Number of queries to generate for this notion, capped at 10
                for _ in range(qper_notion):
                    qlen = random.randint(1, qmax_len)
                    samples = random.sample(all_links, min(qlen, len(all_links)))
                    if len(samples) > 0:
                        query = query_head
                        for link in samples:
                            query_generated = self.generate_graph_query(link, i)
                            if len(query_generated) > 0:
                                query += query_generated
                            if query not in graph_queries["queries"]:
                                graph_queries["queries"].append(query)
        return graph_queries

    # debug method
    def debug(self):
        """Debug method"""
        pass
