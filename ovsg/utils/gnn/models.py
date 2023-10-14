import torch
from torch_geometric.data import Data
from torch.nn import functional as F
from ovsg.utils.gnn.layers import GraphConvolution, GraphAggregator, GraphMatchingConvolution, GraphHeadSimilarity
from ovsg.utils.gnn.utils import adj_matrix_to_edge_index
from ovsg.utils.gnn.utils import create_batch, trim_feats
from ovsg.utils.gnn.utils import acc_f1
from ovsg.utils.gnn.utils import dynamic_partition


class GenericGNN(torch.nn.Module):
    """Generic GNN model"""

    def __init__(self, args):
        super(GenericGNN, self).__init__()
        self.args = args
        if args.n_classes > 2:
            self.f1_average = "micro"
        else:
            self.f1_average = "binary"
        self.layers = torch.nn.ModuleList()
        self.layers.append(GraphConvolution(self.args.feat_dim, self.args.dim, args))
        for _ in range(self.args.num_layers - 1):
            self.layers.append(
                GraphConvolution(self.args.dim, self.args.dim, args),
            )
        self.aggregator = GraphAggregator(self.args.dim, self.args.dim, self.args)
        self.cls = torch.nn.Linear(self.args.dim, 1)

    def compute_emb(self, feats, adjs, sizes):
        """Compute the embedding of the graph"""
        edge_index = adj_matrix_to_edge_index(adjs)
        batch = create_batch(sizes)
        feats = trim_feats(feats, sizes)
        for i in range(self.args.num_layers):
            # convolution
            feats, edge_index = self.layers[i](feats, edge_index)
        # aggregator
        feats = self.aggregator(feats, edge_index, batch)
        return feats

    def forward(self, feats_1, adjs_1, feats_2, adjs_2, sizes_1, sizes_2):
        """Forward pass"""
        # computing the embedding
        emb_1 = self.compute_emb(feats_1, adjs_1, sizes_1)
        emb_2 = self.compute_emb(feats_2, adjs_2, sizes_2)
        outputs = torch.cat((emb_1, emb_2), 1)
        outputs = outputs.reshape(outputs.size(0), -1)

        # classification
        outputs = self.cls.forward(outputs)
        return outputs

    def compute_metrics(self, outputs, labels, split, backpropagate=False):
        """Compute the metrics"""
        outputs = F.log_softmax(outputs, dim=1)
        loss = F.nll_loss(outputs, labels)
        if backpropagate:
            loss.backward()
        if split == "train":
            verbose = True
        else:
            verbose = True
        acc, f1 = acc_f1(
            outputs,
            labels,
            average=self.f1_average,
            logging=self.args.logging,
            verbose=verbose,
        )
        metrics = {"loss": loss, "acc": acc, "f1": f1}
        return metrics, outputs.shape[0]

    def init_metric_dict(self):
        """Initialize the metric dictionary"""
        return {"acc": -1, "f1": -1}

    def has_improved(self, m1, m2):
        """Check if the model has improved"""
        return m1["acc"] < m2["acc"]


class GraphMatchingNetwork(torch.nn.Module):
    """Graph Matching Network, ICLR 2019"""

    def __init__(self, args):
        super(GraphMatchingNetwork, self).__init__()
        self.args = args
        self.layers = torch.nn.ModuleList()
        self.layers.append(GraphMatchingConvolution(self.args.feat_dim, self.args.dim, args))
        for _ in range(self.args.num_layers - 1):
            self.layers.append(GraphMatchingConvolution(self.args.dim, self.args.dim, args))
        self.aggregator = GraphAggregator(self.args.dim, self.args.dim, self.args)
        self.sim = GraphHeadSimilarity()

    def compute_emb(self, feats, edge_index, batch):
        """Compute the embedding of the graph"""
        # data = Data(x=feats, edge_index=edge_index, batch=batch)
        for i in range(self.args.num_layers):
            # convolution
            feats, edge_index, batch = self.layers[i](feats, edge_index, batch)
        return feats, edge_index, batch

    def combine_pair_embedding(
        self, feats_1, feats_2, edge_index_1, edge_index_2, batch_1, batch_2, n_graphs
    ):
        """Pair embedding"""
        feats = torch.cat([feats_1, feats_2], dim=0)
        edge_index = torch.cat([edge_index_1, edge_index_2], dim=1)
        batch = torch.cat([batch_1, batch_2 + n_graphs], dim=0)
        feats = feats.to(self.args.device)
        edge_index = edge_index.to(self.args.device)
        batch = batch.to(self.args.device)
        return feats, edge_index, batch

    def forward(self, x_s, x_t, edge_index_s, edge_index_t, x_s_batch, x_t_batch, num_graphs):
        """Forward pass"""
        # For GMN, we first compute the pair embedding
        # computing the embedding
        feats, edge_index, batch = self.combine_pair_embedding(
            x_s,
            x_t,
            edge_index_s,
            edge_index_t,
            x_s_batch,
            x_t_batch,
            num_graphs,
        )
        feats, edge_index, batch = self.compute_emb(feats, edge_index, batch)
        dist = self.sim(feats, edge_index, batch)
        return dist

    def compute_metrics(self, outputs, labels, split, backpropagate=False):
        """Compute the metrics"""
        labels = labels.to(self.args.device).reshape([-1, 1])
        # binary cross entropy
        loss = F.binary_cross_entropy_with_logits(outputs, labels)
        # binary preds
        preds = (outputs > 0).long()
        if backpropagate:
            loss.backward()
        if split == "train":
            verbose = True
        else:
            verbose = True
        metrics = {"loss": loss, "preds": preds}
        return metrics, outputs.shape[0]


class GraphSimKernelModel(torch.nn.Module):
    """Graph Similarity using Kernel"""

    def __init__(self, kernel_method: str = "head", **kwargs):
        super(GraphSimKernelModel, self).__init__()
        self.kernel_method = kernel_method
        if kernel_method == "head":
            self.sim_func = self.head
        elif kernel_method == "jaccard":
            self.sim_func = self.jaccard
        elif kernel_method == "szymkiewicz_simpson":
            self.sim_func = self.szymkiewicz_simpson
        else:
            raise NotImplementedError
        self.threshold = kwargs.get("threshold", 0.5)

    def forward(self, x_s, x_t, edge_index_s, edge_index_t, batch_s, batch_t, n_graphs):
        """Forward pass"""
        results = [None for _ in range(n_graphs)]
        x_s_partition = dynamic_partition(x_s, batch_s, n_graphs)
        x_t_partition = dynamic_partition(x_t, batch_t, n_graphs)

        for i in range(0, n_graphs):
            sim = self.sim_func(x_s_partition[i], x_t_partition[i])
            results[i] = 1.0 - sim  # dist
        results = torch.cat(results, dim=0)
        results = results.view([n_graphs, 1])
        return results

    def head(self, x_s, x_t):
        return torch.dot(x_s[0, :], x_t[0, :]).reshape([1, 1])

    def jaccard(self, x_s, x_t):
        """Check this: https://en.wikipedia.org/wiki/Jaccard_index"""
        # compute the intersection
        sim_m = torch.mm(x_s, x_t.t())
        # only left the largest in each col
        _, max_indices = torch.max(sim_m, dim=0)
        mask = torch.zeros_like(sim_m)
        for col, row in enumerate(max_indices):
            mask[row, col] = 1
        sim_m = sim_m * mask
        sim_m_max = torch.max(sim_m, dim=1, keepdim=True)[0]
        # computer intersection
        intersection = torch.sum(sim_m_max > self.threshold)
        jaccard_index = intersection / (x_s.shape[0] + x_t.shape[0] - intersection)
        return jaccard_index.reshape([1, 1])

    def szymkiewicz_simpson(self, x_s, x_t):
        """Check this: https://en.wikipedia.org/wiki/Overlap_coefficient"""
        # compute the intersection
        sim_m = torch.mm(x_s, x_t.t())
        # only left the largest in each col
        _, max_indices = torch.max(sim_m, dim=0)
        mask = torch.zeros_like(sim_m)
        for col, row in enumerate(max_indices):
            mask[row, col] = 1
        sim_m = sim_m * mask
        sim_m_max = torch.max(sim_m, dim=1, keepdim=True)[0]
        # computer intersection
        intersection = torch.sum(sim_m_max > self.threshold)
        szymkiewicz_simpson_index = intersection / min(x_s.shape[0], x_t.shape[0])
        return szymkiewicz_simpson_index.reshape([1, 1])
