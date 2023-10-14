from __future__ import annotations
import numpy as np
import torch
import heapq
import abc
from abc import ABC, abstractmethod
import cv2
from PIL import Image, ImageDraw, ImageFont
import networkx as nx
import matplotlib.pyplot as plt
import open3d as o3d
from enum import Enum
from ovsg.env.algo.region import Region
import copy


class Space(Enum):
    """Space of the notion in the NotionGraph"""

    VIRTUAL = 0  # notion existing in virtual space
    STATIC = 1  # notion existing in physical space
    DYNAMIC = 2  # notion moving in physical space
    REGION = 3  # notion represents in a region in physical space


class Domain(Enum):
    """Domain of the notion in the NotionGraph"""

    UKN = 0  # Unknown domain
    USR = 1  # User domain
    TAG = 2  # Tag domain
    INS = 3  # Instance domain
    RGN = 4  # Region domain


class FeatureType(Enum):
    """Feature type of the notion in the NotionGraph"""

    ONE = 0  # feature whose similarity is always 1
    WV = 1  # word vector text feature
    ST = 2  # sentence transformer text feature
    CLIPTXT = 3  # clip text feature
    CLIPIMG = 4  # clip image feature
    DETICIMG = 5  # detic image feature
    SPATIAL = 6  # spatial feature


class Feature:
    """Feature Class"""

    def __init__(self, feature: np.ndarray, feature_type: FeatureType):
        self.feature = feature
        self.feature_type = feature_type

    def sprob(self, query: "Feature"):
        """Probability of self and query being the same"""
        if self.feature_type == FeatureType.ONE or query.feature_type == FeatureType.ONE:
            return 1.0
        elif self.feature_type == FeatureType.WV and query.feature_type == FeatureType.WV:
            feature1_normalized = self.feature / (np.linalg.norm(self.feature) + 1e-8)
            feature2_normalized = query.feature / (np.linalg.norm(query.feature) + 1e-8)
            cos_sim = np.dot(feature1_normalized, feature2_normalized)
            # WordVec similarity is between -1 and 1
            return (cos_sim + 1.0) / 2.0
        elif self.feature_type == FeatureType.ST and query.feature_type == FeatureType.ST:
            feature1_normalized = self.feature / (np.linalg.norm(self.feature) + 1e-8)
            feature2_normalized = query.feature / (np.linalg.norm(query.feature) + 1e-8)
            cos_sim = np.dot(feature1_normalized, feature2_normalized)
            # STTXT similarity is between 0 and 1
            if cos_sim == 0.0:
                return 0.5  # meaning this is a preserved keyword
            return cos_sim
        elif (self.feature_type, query.feature_type) in [
            (FeatureType.CLIPTXT, FeatureType.CLIPTXT),
            (FeatureType.CLIPIMG, FeatureType.CLIPIMG),
            (FeatureType.CLIPTXT, FeatureType.CLIPIMG),
            (FeatureType.CLIPIMG, FeatureType.CLIPTXT),
        ]:
            # CLIP similarity is between -1 and 1
            feature1_normalized = self.feature / (np.linalg.norm(self.feature) + 1e-8)
            feature2_normalized = query.feature / (np.linalg.norm(query.feature) + 1e-8)
            cos_sim = np.dot(feature1_normalized, feature2_normalized)
            return (cos_sim + 1.0) / 2.0
        elif (self.feature_type, query.feature_type) in [
            (FeatureType.DETICIMG, FeatureType.CLIPTXT),
            (FeatureType.CLIPTXT, FeatureType.DETICIMG),
        ]:
            # CLIP & DETIC similarity is between -1 and 1 and ususally low
            feature1_normalized = self.feature / (np.linalg.norm(self.feature) + 1e-8)
            feature2_normalized = query.feature / (np.linalg.norm(query.feature) + 1e-8)
            cos_sim = np.dot(feature1_normalized, feature2_normalized)
            # amplify the similarity between -0.2 ~ 0.2 to -0.8 ~ 0.8.
            if cos_sim > 0.0:
                if cos_sim < 0.2:
                    cos_sim = cos_sim * 4.0
                else:
                    cos_sim = 0.8 + (cos_sim - 0.2) / 4.0
            else:
                if cos_sim > -0.2:
                    cos_sim = cos_sim * 4.0
                else:
                    cos_sim = -0.8 + (cos_sim + 0.2) / 4.0
            # convert to 0 ~ 1
            return (cos_sim + 1.0) / 2.0
        else:
            return 0.0


class User:
    """User Class"""

    def __init__(self, names: list[str], photo: np.ndarray):
        self.names = names
        self.photo = photo


# Predefined notion constants
NotionConstants = {
    "Keywords": ["SOMETHING", "SOMEONE", "SOMEWHERE"],
}


class Notion(ABC):
    """Page in Phsical World Web"""

    def __init__(
        self,
        address: np.ndarray,
        domain: Domain,
        keys: list[Feature],
        content: any,
        forward_links: list[int],
        backward_links: list[int],
        name: str,  # readable name of the notion,
        space: Space = Space.VIRTUAL,
        **kwargs,
    ):
        """Initialize a notion in Physical World Web"""
        self.id = None  # should be set by NotionGraph
        self.address = address
        self.domain = domain
        self.keys = keys
        self.forward_links = forward_links
        self.backward_links = backward_links
        self.content = content
        self.name = name
        self.space = space
        self.expire = False
        # other attributes
        self.attributes = kwargs

    def __hash__(self):
        return hash(self.id)

    def __eq__(self, other):
        return self.id == other.id

    def to_dict(self):
        """Convert to dict"""
        return {
            "id": self.id,
            "address": self.address,
            "domain": self.domain.name,
            "forward_links": self.forward_links,
            "backward_links": self.backward_links,
            "name": self.name,
            "space": self.space.name,
            "expire": self.expire,
            "attributes": self.attributes,
        }

    def sprob(self, query: "Notion" | Feature | list[Feature]):
        """Same probability between two notions"""
        if isinstance(query, Feature):
            # here query is a feature
            max_sprob = 0.0
            for key in self.keys:
                max_sprob = max(max_sprob, key.sprob(query))
            return max_sprob
        elif isinstance(query, list):
            # here query is a list of features
            max_sprob = 0.0
            for key in self.keys:
                for qkey in query:
                    max_sprob = max(max_sprob, key.sprob(qkey))
            return max_sprob
        elif isinstance(query, Notion):
            # here query is a notion
            if self.domain != query.domain:
                # only notions in the same domain can be compared
                return 0.0
            else:
                max_sprob = 0.0
                for key in self.keys:
                    for qkey in query.keys:
                        max_sprob = max(max_sprob, key.sprob(qkey))
                return max_sprob
        else:
            raise NotImplementedError

    @abstractmethod
    def view(self):
        """View the content of the notion"""
        raise NotImplementedError

    def link(self, notion):
        """Link to another notion"""
        self.forward_links.append(notion.id)
        notion.backward_links.append(self.id)

    def unlink(self, notion):
        """Unlink to another notion"""
        self.forward_links.remove(notion.id)
        notion.backward_links.remove(self.id)

    @property
    def links(self):
        """Return all links of the notion"""
        return [*self.forward_links, *self.backward_links]


class NotionTxt(Notion):
    """NotionGraph Page representing a text"""

    def __init__(
        self,
        address: np.ndarray,
        domain: Domain,
        keys: list[Feature],
        content: str,
        forward_links: list[int],
        backward_links: list[int],
        name: str,  # readable name of the notion,
        space: Space = Space.VIRTUAL,
        **kwargs,
    ):
        """Initialize a notion in Physical World Web"""
        super().__init__(
            address, domain, keys, content, forward_links, backward_links, name, space, **kwargs
        )

    def view(self):
        """View the content of the notion"""
        show_name = self.name + "@" + self.domain.name + "." + self.space.name
        print("Viewing notion: {}".format(show_name))
        print(f"Page: {show_name}, Content: {self.content}")
        cv2.waitKey(0)


class NotionImg(Notion):
    """NotionGraph Page representing an image"""

    def __init__(
        self,
        address: np.ndarray,
        domain: Domain,
        keys: list[Feature],
        content: np.ndarray | Image.Image,
        forward_links: list[int],
        backward_links: list[int],
        name: str,  # readable name of the notion
        space: Space = Space.VIRTUAL,
        **kwargs,
    ):
        """Initialize a notion in Physical World Web"""
        super().__init__(
            address, domain, keys, content, forward_links, backward_links, name, space, **kwargs
        )

    def view(self):
        """View the content of the notion"""
        show_name = (
            (self.name + "@" + self.domain.name + "." + self.space.name)
            if self.name
            else (self.address + "@" + self.domain.name + "." + self.space.name)
        )
        if isinstance(self.content, np.ndarray):
            cv2.imshow(show_name, self.content)
            cv2.waitKey(0)
            cv2.destroyAllWindows()
        elif isinstance(self.content, Image.Image):
            # PIL show is not working, due to parallel processing
            np_img = np.array(self.content)
            cv2.imshow(show_name, np_img)
            cv2.waitKey(0)
            cv2.destroyAllWindows()


class NotionPcd(Notion):
    """NotionGraph Page representing a point cloud"""

    def __init__(
        self,
        address: np.ndarray,
        domain: Domain,
        keys: list[Feature],
        content: o3d.geometry.PointCloud,
        forward_links: list[int],
        backward_links: list[int],
        name: str,  # readable name of the notion
        space: Space = Space.VIRTUAL,
        **kwargs,
    ):
        """Initialize a notion in Physical World Web"""
        super().__init__(
            address, domain, keys, content, forward_links, backward_links, name, space, **kwargs
        )

    def view(self):
        """View the content of the notion"""
        # show_name = (
        #     (self.name + "@" + self.domain) if self.name else (self.address + "@" + self.domain)
        # )
        o3d.visualization.draw_geometries([self.content])


class NotionUser(Notion):
    """NotionGraph Page representing a user"""

    def __init__(
        self,
        address: np.ndarray,
        domain: Domain,
        keys: list[Feature],
        content: User,
        forward_links: list[int],
        backward_links: list[int],
        name: str,  # readable name of the notion
        space: Space = Space.VIRTUAL,
        **kwargs,
    ):
        """Initialize a notion in Physical World Web"""
        super().__init__(
            address, domain, keys, content, forward_links, backward_links, name, space, **kwargs
        )

    def view(self):
        """View the content of the notion"""
        show_name = self.name + "@" + self.domain.name + "." + self.space.name
        # resize the image height to 500
        height, width = self.content.photo.shape[:2]
        ratio = 500 / height
        img_vis = cv2.resize(self.content.photo, (int(width * ratio), 500))
        cv2.imshow(show_name, img_vis)
        cv2.waitKey(0)
        cv2.destroyAllWindows()


class NotionRegion(Notion):
    """NotionGraph Page representing a region"""

    def __init__(
        self,
        address: np.ndarray,
        domain: Domain,
        keys: list[Feature],
        content: Region,
        forward_links: list[int],
        backward_links: list[int],
        name: str,  # readable name of the notion
        space: Space = Space.VIRTUAL,
        **kwargs,
    ):
        """Initialize a notion in Physical World Web"""
        super().__init__(
            address, domain, keys, content, forward_links, backward_links, name, space, **kwargs
        )

    def view(self):
        """View the content of the notion"""
        show_name = self.name + "@" + self.domain.name + "." + self.space.name
        self.content.visualize(title=show_name)


class NotionLink(ABC):
    """Link in Physical World Web"""

    def __init__(self, id1: int, id2: int, notion_name1: str, notion_name2: str):
        self.id1 = id1
        self.id2 = id2
        self.notion_name1 = notion_name1
        self.notion_name2 = notion_name2
        self.relation_key = []
        self.relation_desc = []
        self.relation_confid = []

    def add_relation(self, key: Feature, description: str, confidence: float):
        """Add relation between two notions"""
        self.relation_key.append(key)
        self.relation_desc.append(description)
        self.relation_confid.append(confidence)

    def view(self):
        """View the content of the link"""
        print(f"Viewing link: {self.notion_name1} ({self.id1}) -> {self.notion_name2} ({self.id2})")
        for relation in self.relation_desc:
            print(relation)

    def __hash__(self):
        return hash((self.id1, self.id2))

    def __eq__(self, other):
        return (self.id1, self.id2) == (other.id1, other.id2)


class NotionGraph:
    """NotionGraph"""

    _links: dict[tuple[int, int], NotionLink]
    _notions: list[Notion]
    _notion_domain: dict[str, list[int]]
    _notion_space: dict[Space, list[int]]
    _notion_count: int

    def __init__(self, **kwargs):
        self._notions = []  # Store all notions in NotionGraph
        self._links = {}  # Store all links in NotionGraph
        self._notion_domain = {}  # Store all notions in NotionGraph, grouped by domain
        self._notion_space = {}  # Store all notions in NotionGraph, grouped by space
        self._notion_count = 0

    def reset(self):
        """Reset NotionGraph"""
        self._notions = []
        self._links = {}
        self._notion_domain = {}
        self._notion_space = {}
        self._notion_count = 0

    def empty(self):
        """Check if NotionGraph is empty"""
        return len(self._notions) == 0

    def add_notion(
        self,
        keys: list[Feature],
        notion: any,
        address: np.ndarray,
        domain: Domain,
        name: str,
        space: Space = Space.VIRTUAL,
        **kwargs,
    ) -> int:
        """Add notion to NotionGraph"""
        if isinstance(notion, Notion):
            # update notion
            notion.id = self._notion_count
            self._notions.append(notion)
            self._notion_count += 1
            # update domain
            if domain not in self._notion_domain:
                self._notion_domain[domain] = []
            self._notion_domain[domain].append(notion.id)
            # update space
            if space not in self._notion_space:
                self._notion_space[space] = []
            self._notion_space[space].append(notion.id)
            return len(self._notions) - 1
        else:
            raise ValueError("Page must be an instance of Notion")

    def _build_notion(
        self,
        key: list[np.ndarray] | None,
        content: any,
        address: np.ndarray,
        domain: Domain,
        name: str,
        space: Space = Space.VIRTUAL,
        **kwargs,
    ) -> Notion:
        if key is None:
            raise ValueError("Key cannot be None")
        else:
            if isinstance(content, o3d.geometry.PointCloud):
                return NotionPcd(address, domain, key, content, [], [], name, space, **kwargs)
            elif isinstance(content, Region):
                return NotionRegion(address, domain, key, content, [], [], name, space, **kwargs)
            elif isinstance(content.User):
                return NotionUser(address, domain, key, content, [], [], name, space, **kwargs)
            else:
                raise ValueError("Content type not supported")

    # interface methods
    def link(
        self,
        notion1: Notion | int,
        relation: str,
        notion2: Notion | int,
        confidence: float = 1.0,
        relation_keys: list[Feature] | None = None,
    ):
        """Link two notions in NotionGraph"""
        raise NotImplementedError

    def unlink(self, notion1: Notion, notion2: Notion):
        """Unlink two notions in NotionGraph"""
        raise NotImplementedError

    def update(self, **kwargs):
        """update NotionGraph"""
        raise NotImplementedError

    # Search methods
    def search(
        self, query: str, top_k: int, domain: Domain, space: Space, **kwargs
    ) -> tuple[list[Notion], list[float]]:
        """Search query in NotionGraph"""
        raise NotImplementedError

    def search_by_key(
        self,
        query: list[np.ndarray],
        top_k: int,
        domain: Domain | None,
        space: Space | None,
        **kwargs,
    ) -> tuple[list[Notion], list[float]]:
        """Search key in NotionGraph, enable multiple keys"""
        # use a heap to store top k results
        # create a list of tuples (notion, similarity) using the sim function
        notion_sprobs = [
            (notion, notion.sprob(query)) for notion in self.at(domain=domain, space=space)
        ]
        if top_k > 0 and top_k <= len(notion_sprobs):
            top_k_notion_sprobs = heapq.nlargest(top_k, notion_sprobs, key=lambda x: x[1])
            top_k_notions, top_k_sprobs = zip(*top_k_notion_sprobs)
            return list(top_k_notions), list(top_k_sprobs)
        else:
            # return all sorted
            notion_sprobs.sort(key=lambda x: x[1], reverse=True)
            notions, sprobs = zip(*notion_sprobs)
            return list(notions), list(sprobs)

    def _subgraph_notions(self, id: int, depth: int) -> list[int]:
        """Get subgraph of a notion of depth"""
        if depth == 0:
            return [id]
        else:
            subgraph_id = [id]
            for linked_id in self._notions[id].links:
                subgraph_id += self._subgraph_notions(linked_id, depth - 1)
            # remove duplicates while preserving order
            # https://stackoverflow.com/questions/480214/how-do-you-remove-duplicates-from-a-list-whilst-preserving-order
            return list(dict.fromkeys(subgraph_id))

    def subgraph_nodelink(
        self, notion: Notion | int, depth: int = 1
    ) -> tuple[list[Notion], list[NotionLink]]:
        """Get subgraph of a notion of depth"""
        if isinstance(notion, Notion):
            id = notion.id
        elif isinstance(notion, int):
            id = notion
            notion = self._notions[id]
        else:
            raise ValueError("Page must be an instance of Notion or int")
        # notions id
        subgraph_id = self._subgraph_notions(id, depth)
        # links
        # only log links between notion and root notion
        subgraph_links = []
        for id in subgraph_id:
            if id in notion.forward_links:
                subgraph_links.append(self.links[(notion.id, id)])
            if id in notion.backward_links:
                subgraph_links.append(self.links[(id, notion.id)])

        return [self._notions[id] for id in subgraph_id], list(set(subgraph_links))

    def subgraph(self, notion: Notion | int, depth: int = 1) -> "NotionGraph":
        """Get subgraph of a notion of depth"""
        notions, links = self.subgraph_nodelink(notion, depth)
        graph = NotionGraph()
        old2new = {}  # id mapping
        for i, notion_ in enumerate(notions):
            old2new[notion_.id] = i
        # update notions
        graph._notions = []
        for notion_ in notions:
            new_notion = copy.deepcopy(notion_)
            new_notion.id = old2new[notion_.id]
            # update notion links
            new_forward_links = []
            for i, id in enumerate(notion_.forward_links):
                if id in old2new:
                    new_forward_links.append(old2new[id])
            new_notion.forward_links = new_forward_links
            new_backward_links = []
            for i, id in enumerate(notion_.backward_links):
                if id in old2new:
                    new_backward_links.append(old2new[id])
            new_notion.backward_links = new_backward_links
            graph._notions.append(new_notion)
        # update links
        graph._links = {}
        for link_ in links:
            new_link = copy.deepcopy(link_)
            new_link.id1 = old2new[link_.id1]
            new_link.id2 = old2new[link_.id2]
            graph._links[(new_link.id1, new_link.id2)] = new_link
        return graph

    # generator
    def __iter__(self):
        for notion in self._notions:
            yield notion

    def at(self, domain:  Domain | None = None, space: Space | None = None):
        """Get all notions"""
        if domain is None and space is None:
            for notion in self._notions:
                yield notion
        elif domain is not None and space is None:
            for notion in self.at_domain(domain):
                yield notion
        elif domain is None and space is not None:
            for notion in self.at_space(space):
                yield notion
        else:
            for notion in self.at_domain_space(domain, space):
                yield notion

    def at_domain(self, domain: Domain):
        """Get all notions at a domain"""
        if domain not in self._notion_domain:
            return []
        for id in self._notion_domain[domain]:
            yield self._notions[id]

    def at_space(self, space: Space):
        """Get all notions at a space"""
        if space not in self._notion_space:
            return []
        for id in self._notion_space[space]:
            yield self._notions[id]

    def at_domain_space(self, domain: Domain, space: Space):
        """Get all notions at a domain and space"""
        if domain not in self._notion_domain:
            return []
        if space not in self._notion_space:
            return []
        for id in self._notion_domain[domain]:
            if id in self._notion_space[space]:
                yield self._notions[id]

    # properties
    @property
    def notions(self):
        """Get notions"""
        return self._notions

    @notions.setter
    def notions(self, notions):
        """Pages cannot be set directly. Use add_notion() instead."""
        raise ValueError("Pages cannot be set directly. Use add_notion() instead.")

    @property
    def links(self):
        """Get links"""
        return self._links

    @links.setter
    def links(self, links):
        """Links cannot be set directly. Use link() instead."""
        raise ValueError("Links cannot be set directly. Use link() instead.")

    @property
    def notion_count(self):
        """Get notion count"""
        return self._notion_count

    @notion_count.setter
    def notion_count(self, notion_count):
        """notion count is read-only"""
        raise ValueError("Page count cannot be set directly.")

    @property
    def notion_domain(self):
        """Get all notions at a domain"""
        return self._notion_domain

    @notion_domain.setter
    def notion_domain(self, notion_domain):
        raise ValueError("Pages domain cannot be set directly.")

    @property
    def total_size(self):
        """Get total size"""
        return len(self._notions) + len(self._links)

    # debug method
    def ping(self, notion1:  int | Notion, notion2: int | Notion):
        """notion1 ping notion2, checking if notion1 are linked with notion2"""
        if isinstance(notion1, Notion):
            notion1_id = notion1.id
        elif isinstance(notion1, int):
            notion1_id = notion1
        else:
            raise ValueError("Page must be an instance of Notion or int")
        if isinstance(notion2, Notion):
            notion2_id = notion2.id
        elif isinstance(notion2, int):
            notion2_id = notion2
        else:
            raise ValueError("Page must be an instance of Notion or int")

        success = False
        if (notion1_id, notion2_id) in self._links:
            link = self._links[(notion1_id, notion2_id)]
            link.view()
            success = True

        if (notion2_id, notion1_id) in self._links:
            link = self._links[(notion2_id, notion1_id)]
            link.view()
            success = True
        return success

    def visualize(self, **kwargs):
        """Visualize NotionGraph"""
        G = nx.DiGraph()
        node_icons = []
        for notion in self._notions:
            label = notion.name + "@" + notion.domain.name
            G.add_node(notion.id, label=label)
            if isinstance(notion, NotionImg):
                # resize it to 100x100 (PIL image)
                img = notion.content.resize((100, 100))
                node_icons.append(img)
            elif isinstance(notion, NotionTxt):
                node_icons.append(notion.content)
            else:
                node_icons.append(None)

        for link in self._links.values():
            label = link.relation_desc[0] if link.relation_desc else ""
            G.add_edge(link.id1, link.id2, label=label)

        # define a function to display the images on the nodes
        def draw_image(image, pos, ax=None):
            if ax is None:
                ax = plt.gca()
            im = ax.imshow(image, extent=(pos[0] - 0.2, pos[0] + 0.2, pos[1] - 0.2, pos[1] + 0.2))
            return im

        # visualize the graph with images on nodes
        pos = nx.spring_layout(G)
        fig, ax = plt.subplots(figsize=(5, 5))
        # nx.draw_networkx_nodes(G, pos=pos, node_size=500, node_color="w", edgecolors="k", ax=ax)
        for i in range(len(node_icons)):
            if isinstance(node_icons[i], Image.Image):
                draw_image(node_icons[i], pos[i], ax=ax)
            if isinstance(node_icons[i], str):
                # thicker font
                ax.text(pos[i][0], pos[i][1], node_icons[i], fontsize=20, fontweight="bold")
        nx.draw_networkx_edges(G, pos=pos, ax=ax)
        nx.draw_networkx_edge_labels(G, pos=pos)  # add edge labels
        plt.axis("off")
        # set limit
        plt.xlim(-1.2, 1.2)
        plt.ylim(-1.2, 1.2)
        plt.show()

    # utility methods
    def encode_text(self, text: str, **kwargs):
        """Encode text to vector"""
        raise NotImplementedError


class NotionGraphWrapper(ABC):
    """Wrapper for NotionGraph"""

    def __init__(self, notion_graph: NotionGraph, **kwargs) -> None:
        """Wrapper for NotionGraph"""
        self.notion_graph = notion_graph

    def reset(self):
        """Reset NotionGraph"""
        self.notion_graph.reset()

    def empty(self):
        """Check if NotionGraph is empty"""
        return self.notion_graph.empty()

    def add_notion(
        self,
        keys: list[Feature],
        notion: any,
        address: np.ndarray,
        domain: Domain,
        name: str,
        space: Space = Space.VIRTUAL,
        **kwargs,
    ) -> int:
        """Add a notion to NotionGraph"""
        return self.notion_graph.add_notion(keys, notion, address, domain, name, space, **kwargs)

    def link(
        self,
        notion1:  Notion | int,
        relation: str,
        notion2:  Notion | int,
        confidence: float = 1.0,
        relation_keys:  list[Feature] | None = None,
    ):
        """Link two notions with a relation"""
        self.notion_graph.link(notion1, relation, notion2, confidence)

    def unlink(self, notion1: Notion, notion2: Notion):
        """Unlink two notions"""
        self.notion_graph.unlink(notion1, notion2)

    def update(self, **kwargs):
        """Update NotionGraph"""
        self.notion_graph.update(**kwargs)

    def search(
        self,
        query: str,
        top_k: int,
        domain:  Domain | None = None,
        space:  Space | None = None,
        **kwargs,
    ) -> tuple[list[Notion], list[float]]:
        """Search for notions with a query"""
        return self.notion_graph.search(query, top_k, domain, space, **kwargs)

    def search_by_key(
        self,
        query: list[np.ndarray],
        top_k: int,
        domain:   Domain | None,
        space:  Space | None,
    ) -> tuple[list[Notion], list[float]]:
        """Search for notions with a query"""
        return self.notion_graph.search_by_key(query, top_k, domain=domain, space=space)

    def subgraph_nodelink(
        self, notion:  Notion | int, depth: int = 1
    ) -> tuple[list[Notion], list[NotionLink]]:
        """Get subgraph_nodelink of a notion"""
        return self.notion_graph.subgraph_nodelink(notion, depth)

    def subgraph(self, notion:  Notion | int, depth: int = 1) -> NotionGraph:
        """Get subgraph of a notion"""
        return self.notion_graph.subgraph(notion, depth)

    def ping(self, notion1:  Notion | int, notion2:  Notion | int):
        """Ping a notion"""
        self.notion_graph.ping(notion1, notion2)

    def visualize(self, **kwargs):
        """Visualize NotionGraph"""
        self.notion_graph.visualize(**kwargs)

    # generator methods
    def __iter__(self):
        return self.notion_graph.__iter__()

    def at(self, domain:  Domain | None = None, space:  Space | None = None):
        """Iterate over notions"""
        return self.notion_graph.at(domain=domain, space=space)

    # properties
    @property
    def notions(self):
        """Get all notions"""
        return self.notion_graph.notions

    @property
    def links(self):
        """Get all links"""
        return self.notion_graph.links

    @property
    def notion_count(self):
        """Get notion count"""
        return self.notion_graph.notion_count

    @property
    def notion_domain(self):
        """Get all domains"""
        return self.notion_graph.notion_domain

    @property
    def total_size(self):
        """Get total size"""
        return self.notion_graph.total_size

    # utility methods
    def encode_text(self, text: str, **kwargs) -> np.ndarray:
        """Encode text to vector"""
        return self.notion_graph.encode_text(text, **kwargs)

    # utility only for wrapper
    def generate(self, target: str, **kwargs):
        """Generate something"""
        raise NotImplementedError
