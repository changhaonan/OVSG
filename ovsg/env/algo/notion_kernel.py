from __future__ import annotations
from ovsg.env.algo.notion import NotionGraphWrapper, NotionGraph, NotionLink
import hydra
import torch
import numpy as np
import networkx as nx
from matplotlib import pyplot as plt


class NotionKernelWrapper(NotionGraphWrapper):
    """Wrapper for notion graph kernel, including gnn, jaccard, szymkiewicz_simpson"""

    def __init__(
        self,
        notion_graph: NotionGraph | NotionGraphWrapper,
        padding_size: int,
        max_node_size: int,
    ):
        self.notion_graph = notion_graph
        self.padding_size = padding_size
        self.max_node_size = max_node_size
        # create spatial encoder
        root_abs_path = hydra.utils.get_original_cwd()
        cfg = hydra.compose(config_name="ml/gnn_matcher")

        # load gnn
        args = GMatchArgs(cfg)
        gnn_model = GraphMatchingNetwork(args).to(cfg.device)
        # load model
        self.gnn_matcher = GraphMatcher(cfg, root_abs_path, gnn_model)
        self.gnn_matcher.load()

        # load kernel model (traditional kernel)
        kernel_model = GraphSimKernelModel("head", threshold=cfg.threshold)
        self.kernel_matcher = GraphMatcher(cfg, root_abs_path, kernel_model)

    # main interface
    def generate(self, target: str, **kwargs):
        """Generate interface"""
        if target == "gnn_triplet":
            assert "G_s" in kwargs and "id_p" in kwargs and "id_n" in kwargs
            G_s = kwargs.pop("G_s")
            id_p = kwargs.pop("id_p")
            id_n = kwargs.pop("id_n")
            return self.generate_triplet(G_s=G_s, id_p=id_p, id_n=id_n, **kwargs)
        elif target == "gnn_pair":
            assert "G_s" in kwargs and "id" in kwargs
            G_s = kwargs.pop("G_s")
            id = kwargs.pop("id")
            return self.generate_pair(G_s=G_s, id=id, **kwargs)
        elif target == "kernel_dist":
            assert "G_s" in kwargs and "candid_ids" in kwargs and "kernel_method" in kwargs
            G_s = kwargs.pop("G_s")
            candid_ids = kwargs.pop("candid_ids")
            kernel_method = kwargs.pop("kernel_method")
            return self.generate_dist(G_s=G_s, candid_ids=candid_ids, kernel_method=kernel_method)
        else:
            # pass it to other wrapper
            return self.notion_graph.generate(target, **kwargs)

    def generate_triplet(self, G_s: NotionGraph, id_p: int, id_n: int, **kwargs):
        """Generate (G_s, G_p, G_n) triplet for training"""
        assert id_p is not id_n
        # get notion graph
        G_p = self.notion_graph.subgraph(id_p)
        G_n = self.notion_graph.subgraph(id_n)
        x_s, x_p, x_n, edge_index_s, edge_index_p, edge_index_n = self.transfer(
            G_s, G_p, G_n, self.padding_size, **kwargs
        )
        return x_s, x_p, x_n, edge_index_s, edge_index_p, edge_index_n

    def generate_pair(self, G_s: NotionGraph, id: int, **kwargs):
        """Generate (G_s, G_p) pair for training"""
        G_t = self.notion_graph.subgraph(id)
        x_s, x_t, edge_index_s, edge_index_t = self.transfer(
            G_s, G_t, None, self.padding_size, **kwargs
        )
        # self.visualize_graph_pair(x_s, x_t, edge_index_s, edge_index_t, **kwargs)
        return x_s, x_t, edge_index_s, edge_index_t

    # GNN-related methods
    def transfer(
        self,
        G_s: NotionGraph,
        G_p: NotionGraph,
        G_n: NotionGraph | None = None,
        padding_size: int = 10,
        **kwargs
    ):
        """Convert Similarity to Torch feature"""
        assert G_s.total_size <= padding_size
        size_s = G_s.total_size
        size_p = G_p.total_size
        x_s = np.zeros([size_s, padding_size])
        x_p = np.zeros([size_p, padding_size])

        # compute similarity (query feats)
        node_size_s = len(G_s.notions)
        node_size_p = len(G_p.notions)
        # digonal to be 1
        x_s[:size_s, :size_s] = np.eye(size_s)

        # compute similarity (candid graph)
        for i, notion_i in enumerate(G_p.notions):
            for j, notion_j in enumerate(G_s.notions):
                x_p[i, j] = notion_i.sprob(notion_j)

        for i, (_, link_i) in enumerate(G_p.links.items()):
            for j, (_, link_j) in enumerate(G_s.links.items()):
                x_p[node_size_p + i, node_size_s + j] = self.link_sprob(link_i, link_j)

        # compute adjacency matrix (query graph)
        edge_index_s = np.zeros([2, 2 * len(G_s.links)])
        for i, (node_ij, link) in enumerate(G_s.links.items()):
            edge_index_s[:, 2 * i] = np.array([node_ij[0], node_size_s + i])
            edge_index_s[:, 2 * i + 1] = np.array([node_size_s + i, node_ij[1]])

        # compute adjacency matrix (candid graph)
        edge_index_p = np.zeros([2, 2 * len(G_p.links)])
        for i, (node_ij, link) in enumerate(G_p.links.items()):
            edge_index_p[:, 2 * i] = np.array([node_ij[0], node_size_p + i])
            edge_index_p[:, 2 * i + 1] = np.array([node_size_p + i, node_ij[1]])

        # to torch
        x_s = torch.Tensor(x_s).to(torch.float32)
        x_p = torch.Tensor(x_p).to(torch.float32)
        edge_index_s = torch.Tensor(edge_index_s).to(torch.long)
        edge_index_p = torch.Tensor(edge_index_p).to(torch.long)

        # neg
        if G_n is not None:
            size_n = G_n.total_size
            x_n = np.zeros([size_n, padding_size])
            node_size_n = len(G_n.notions)
            # compute similarity (neg graph)
            for i, notion_i in enumerate(G_n.notions):
                for j, notion_j in enumerate(G_s.notions):
                    x_n[i, j] = notion_i.sprob(notion_j)

            for i, (_, link_i) in enumerate(G_n.links.items()):
                for j, (_, link_j) in enumerate(G_s.links.items()):
                    x_n[node_size_n + i, node_size_s + j] = self.link_sprob(link_i, link_j)

            # compute adjacency matrix (neg graph)
            edge_index_n = np.zeros([2, 2 * len(G_n.links)])
            for i, (node_ij, link) in enumerate(G_n.links.items()):
                edge_index_n[:, 2 * i] = np.array([node_ij[0], node_size_n + i])
                edge_index_n[:, 2 * i + 1] = np.array([node_size_n + i, node_ij[1]])
            x_n = torch.Tensor(x_n).to(torch.float32)
            edge_index_n = torch.Tensor(edge_index_n).to(torch.long)
            return x_s, x_p, x_n, edge_index_s, edge_index_p, edge_index_n
        else:
            return x_s, x_p, edge_index_s, edge_index_p

    def transfer_v2(
        self,
        G_s: NotionGraph,
        G_p: NotionGraph,
        G_n: NotionGraph | None = None,
        padding_size: int = 10,
        **kwargs
    ):
        """Convert Similarity to Torch feature, without considering link"""
        assert G_s.total_size <= padding_size
        size_s = len(G_s.notions)
        size_p = len(G_p.notions)
        x_s = np.zeros([size_s, padding_size])
        x_p = np.zeros([size_p, padding_size])

        # compute similarity (query feats)
        # digonal to be 1
        x_s[:size_s, :size_s] = np.eye(size_s)

        # compute similarity (candid graph)
        for i, notion_i in enumerate(G_p.notions):
            for j, notion_j in enumerate(G_s.notions):
                x_p[i, j] = notion_i.sprob(notion_j)

        # compute adjacency matrix (query graph)
        edge_index_s = np.zeros([2, len(G_s.links)])
        for i, (node_ij, link) in enumerate(G_s.links.items()):
            edge_index_s[:, i] = np.array([node_ij[0], node_ij[1]])

        # compute adjacency matrix (candid graph)
        edge_index_p = np.zeros([2, len(G_p.links)])
        for i, (node_ij, link) in enumerate(G_p.links.items()):
            edge_index_p[:, i] = np.array([node_ij[0], node_ij[1]])

        # to torch
        x_s = torch.Tensor(x_s).to(torch.float32)
        x_p = torch.Tensor(x_p).to(torch.float32)
        edge_index_s = torch.Tensor(edge_index_s).to(torch.long)
        edge_index_p = torch.Tensor(edge_index_p).to(torch.long)

        # neg
        if G_n is not None:
            size_n = len(G_n.notions)
            x_n = np.zeros([size_n, padding_size])
            # compute similarity (neg graph)
            for i, notion_i in enumerate(G_n.notions):
                for j, notion_j in enumerate(G_s.notions):
                    x_n[i, j] = notion_i.sprob(notion_j)

            # compute adjacency matrix (neg graph)
            edge_index_n = np.zeros([2, len(G_n.links)])
            for i, (node_ij, link) in enumerate(G_n.links.items()):
                edge_index_n[:, i] = np.array([node_ij[0], node_ij[1]])
            x_n = torch.Tensor(x_n).to(torch.float32)
            edge_index_n = torch.Tensor(edge_index_n).to(torch.long)
            return x_s, x_p, x_n, edge_index_s, edge_index_p, edge_index_n
        else:
            return x_s, x_p, edge_index_s, edge_index_p

    def link_sprob(self, link_i: NotionLink, link_j: NotionLink):
        """Compute link similarity"""
        sprob_relation = 0.0
        for relation_desc_i, relation_key_i in zip(link_i.relation_desc, link_i.relation_key):
            for relation_desc_j, relation_key_j in zip(link_j.relation_desc, link_j.relation_key):
                sprob_relation = max(sprob_relation, relation_key_i.sprob(relation_key_j))
                # deal with spatial relation sperately
                if relation_desc_i == "spatial" and relation_desc_j == "spatial":
                    sprob_relation = self.notion_graph.spatial_sprob(relation_key_i, relation_key_j)
        sprob_relation = max(0.0, sprob_relation)  # relation similarity should be positive
        return sprob_relation

    def visualize_graph_pair(
        self,
        x_s: np.ndarray | torch.Tensor,
        x_p: np.ndarray | torch.Tensor,
        edge_index_s: np.ndarray | torch.Tensor,
        edge_index_p: np.ndarray | torch.Tensor,
        **kwargs
    ):
        """Visualize graph"""
        if isinstance(x_s, torch.Tensor):
            x_s = x_s.numpy()
        if isinstance(x_p, torch.Tensor):
            x_p = x_p.numpy()
        if isinstance(edge_index_s, torch.Tensor):
            edge_index_s = edge_index_s.numpy()
        if isinstance(edge_index_p, torch.Tensor):
            edge_index_p = edge_index_p.numpy()

        # predefine a color map
        num_colors = x_s.shape[1]
        colors = np.random.rand(num_colors, 3)  # randomly generate colors
        plt.subplot(121)
        self.visualize_graph(feats=x_s, edge_index=edge_index_s, colors=colors)
        plt.subplot(122)
        self.visualize_graph(feats=x_p, edge_index=edge_index_p, colors=colors)

        title = kwargs.get("title", "")
        plt.suptitle(title)
        plt.show()

    def visualize_graph(self, feats: np.ndarray, edge_index: np.ndarray, colors: np.ndarray):
        """Visualize graph"""
        # edge_index to adj
        adj = np.zeros([feats.shape[0], feats.shape[0]])
        for i in range(edge_index.shape[1]):
            adj[edge_index[0, i], edge_index[1, i]] = 1
        # draw graph
        graph = nx.from_numpy_array(adj)
        pos = nx.spring_layout(graph)
        # assign color
        color_list = []
        for node in graph.nodes():
            feats_idx = np.argmax(feats[node])
            color = colors[feats_idx] * (0.3 + 0.7 * max(feats[node]))
            color_list.append(color)
        nx.draw(graph, pos, node_color=color_list, node_size=500, with_labels=True)

    # batch mehtod
    def generate_dist(self, G_s: NotionGraph, candid_ids: list[int], kernel_method: str):
        """Rerank candidates using GNN"""
        x_s_list = []
        x_t_list = []
        edge_index_s_list = []
        edge_index_t_list = []
        for id in candid_ids:
            x_s, x_t, edge_index_s, edge_index_t = self.generate_pair(G_s, id)
            x_s_list.append(x_s)
            x_t_list.append(x_t)
            edge_index_s_list.append(edge_index_s)
            edge_index_t_list.append(edge_index_t)
        # batchify
        x_s, x_t, edge_index_s, edge_index_t, x_s_batch, x_t_batch, num_graphs = gnn_data2batch(
            x_s_list, x_t_list, edge_index_s_list, edge_index_t_list
        )
        if kernel_method == "gnn":
            dists = (
                self.gnn_matcher.predict(
                    x_s, x_t, edge_index_s, edge_index_t, x_s_batch, x_t_batch, num_graphs
                )
                .detach()
                .cpu()
                .numpy()
            )
        else:
            self.kernel_matcher.set_kernel(kernel_method)
            dists = (
                self.kernel_matcher.predict(
                    x_s, x_t, edge_index_s, edge_index_t, x_s_batch, x_t_batch, num_graphs
                )
                .detach()
                .cpu()
                .numpy()
            )
        return dists
