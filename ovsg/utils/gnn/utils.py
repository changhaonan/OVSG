import torch
from torch.nn import functional as F
from sklearn.metrics import f1_score, accuracy_score


def acc_f1(output, labels, average="binary", logging=None, verbose=True):
    """Computes the accuracy and F1-score between `output` and `labels`"""
    preds = output.max(1)[1].type_as(labels)
    if preds.is_cuda:
        preds = preds.cpu()
        labels = labels.cpu()
    accuracy = accuracy_score(preds, labels)
    f1 = f1_score(preds, labels, average=average)
    return accuracy, f1


def dynamic_partition(data, partitions, num_partitions):
    """Partition data into num_partitions according to partitions"""
    res = []
    for i in range(num_partitions):
        res.append(data[torch.where(partitions == i)])
    return res


def adj_matrix_to_edge_index(adj_matrix, device=None):
    """Convert adjacency matrix to edge index"""
    edge_index = [[], []]
    for i, row in enumerate(adj_matrix.cpu().detach().numpy().tolist()):
        for j, cell_value in enumerate(row[i + 1:]):
            if cell_value == 1:
                edge_index[0].append(i)
                edge_index[1].append(j)
        edge_index[0].append(i)
        edge_index[1].append(i)
    edge_index = torch.tensor(edge_index, dtype=torch.int64)
    if device:
        edge_index = edge_index.to(device)
    return edge_index


def create_batch(sizes):
    """Create batch from sizes"""
    sizes = sizes.tolist()
    sizes = list(map(int, sizes))
    batch = []
    for i, size in enumerate(sizes):
        batch.extend([i] * size)
    batch = torch.tensor(batch, dtype=torch.int64)
    return batch


def trim_feats(feats, sizes):
    """Trim feats according to sizes"""
    stacked_num_nodes = sum(sizes)
    stacked_tree_feats = torch.zeros((stacked_num_nodes, feats.shape[-1]), dtype=torch.float64)
    start_index = 0
    for i, size in enumerate(sizes):
        end_index = start_index + size
        stacked_tree_feats[start_index:end_index, :] = feats[i, :size, :]
        start_index = end_index
    return stacked_tree_feats


def pairwise_cosine_similarity(a, b):
    """pairwise cosine similarity"""
    a_norm = torch.norm(a, dim=1).unsqueeze(-1)
    b_norm = torch.norm(b, dim=1).unsqueeze(-1)
    return torch.matmul(a_norm, b_norm.T)


def compute_crosss_attention(x_i, x_j):
    """Compute cross attention"""
    a = pairwise_cosine_similarity(x_i, x_j)
    a_i = F.softmax(a, dim=1)
    a_j = F.softmax(a, dim=0)
    att_i = torch.matmul(a_i, x_j)
    att_j = torch.matmul(a_j.T, x_i)
    return att_i, att_j


def batch_block_pair_attention(data, batch, n_graphs):
    """Compute attention between blocks in a batch"""
    results = [None for _ in range(n_graphs * 2)]
    partitions = dynamic_partition(data, batch, n_graphs * 2)
    for i in range(0, n_graphs):
        x = partitions[i]
        y = partitions[i + n_graphs]
        attention_x, attention_y = compute_crosss_attention(x, y)
        results[i] = attention_x
        results[i + n_graphs] = attention_y
    results = torch.cat(results, dim=0)
    results = results.view(data.shape)
    return results
