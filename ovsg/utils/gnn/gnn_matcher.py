""" GNN matcher """
from __future__ import annotations
import os
import hydra
import torch
import torch.nn.functional as F
from torch_geometric.data import Data, Dataset, DataLoader
from torch.utils.data import random_split
from ovsg.utils.gnn.models import GraphSimKernelModel


def gnn_data2batch(
    x_s_list: list[torch.Tensor],
    x_t_list: list[torch.Tensor],
    edge_index_s_list: list[torch.Tensor],
    edge_index_t_list: list[torch.Tensor],
):
    """Convert a list of GNN data to a batch of GNN data"""
    x_s = torch.cat(x_s_list, dim=0)
    x_t = torch.cat(x_t_list, dim=0)
    # append offset to edge_index
    offset_s = 0
    offset_t = 0
    sizes_s = []
    sizes_t = []
    for i in range(len(edge_index_s_list)):
        edge_index_s_list[i] += offset_s
        edge_index_t_list[i] += offset_t
        offset_s += x_s_list[i].size(0)
        offset_t += x_t_list[i].size(0)
        sizes_s.append(x_s_list[i].size(0))
        sizes_t.append(x_t_list[i].size(0))
    edge_index_s = torch.cat(edge_index_s_list, dim=1)
    edge_index_t = torch.cat(edge_index_t_list, dim=1)
    # compute batch
    x_s_batch = []
    x_t_batch = []
    for i, (size_s, size_t) in enumerate(zip(sizes_s, sizes_t)):
        x_s_batch.extend([i] * size_s)
        x_t_batch.extend([i] * size_t)
    x_s_batch = torch.tensor(x_s_batch, dtype=torch.long)
    x_t_batch = torch.tensor(x_t_batch, dtype=torch.long)
    num_graphs = len(sizes_s)
    return x_s, x_t, edge_index_s, edge_index_t, x_s_batch, x_t_batch, num_graphs


def prepare_train_file(root_path: str, enable_user: bool) -> list[str]:
    """Prepare the training file"""
    file_list = []
    scene_list = os.listdir(root_path)
    for scene in scene_list:
        if enable_user:
            scene_data_path = os.path.join(root_path, scene, "gnn_matcher")
            scene_data_list = os.listdir(scene_data_path)
        else:
            scene_data_path = os.path.join(root_path, scene, "no_user/gnn_matcher")
            scene_data_list = os.listdir(scene_data_path)
        scene_data_list = [os.path.join(scene_data_path, x) for x in scene_data_list]
        file_list.extend(scene_data_list)
    return file_list


class TripletData(Data):
    """Triplet data, used for GNN matching"""

    def __inc__(self, key, value, *args, **kwargs):
        if key == "edge_index_s":
            return self.x_s.size(0)
        if key == "edge_index_p":
            return self.x_p.size(0)
        if key == "edge_index_n":
            return self.x_n.size(0)
        return super().__inc__(key, value, *args, **kwargs)


class GraphTripletDataset(Dataset):
    """GNN matcher dataset"""

    def __init__(self, root, transform=None, pre_transform=None, pre_filter=None):
        super().__init__(root, transform, pre_transform, pre_filter)
        self.root = root
        self.file_list = prepare_train_file(root_path=root, enable_user=True)

    @property
    def raw_file_names(self) -> list[str]:
        return self.file_list

    @property
    def processed_file_names(self) -> list[str]:
        return self.file_list

    def len(self):
        return len(self.processed_file_names)

    def get(self, idx):
        raw_data = torch.load(os.path.join(self.root, self.raw_file_names[idx]))
        data = TripletData(
            x_s=raw_data["x_s"].to(torch.float32),
            edge_index_s=raw_data["edge_index_s"].to(torch.long),
            x_p=raw_data["x_p"].to(torch.float),
            edge_index_p=raw_data["edge_index_p"].to(torch.long),
            x_n=raw_data["x_n"].to(torch.float),
            edge_index_n=raw_data["edge_index_n"].to(torch.long),
        )
        return data


class GMatchArgs:
    """GMatch args"""

    def __init__(self, cfg):
        self.feat_dim = cfg.feat_dim
        self.dim = cfg.dim
        self.num_layers = cfg.num_layers
        self.device = cfg.device
        self.logging = cfg.logging
        self.threshold = cfg.threshold


class GraphMatcher:
    """Graph matcher"""

    def __init__(self, cfg, root_abs_path, model):
        self.model = model
        self.lr_rate = cfg.lr_rate
        self.epochs = cfg.epochs
        # param
        self.cfg = cfg
        self.root_abs_path = root_abs_path
        self.save_dir = cfg.save_dir

    def train(self, train_loader, val_loader):
        """Train the model"""
        optimizer = torch.optim.Adam(self.model.parameters(), lr=self.lr_rate)
        # train
        for epoch in range(self.epochs):
            loss_epoch = 0.0
            self.model.train()
            for data in train_loader:
                dist_p = self.model(
                    data.x_s,
                    data.x_p,
                    data.edge_index_s,
                    data.edge_index_p,
                    data.x_s_batch,
                    data.x_p_batch,
                    data.num_graphs,
                )
                dist_n = self.model(
                    data.x_s,
                    data.x_n,
                    data.edge_index_s,
                    data.edge_index_n,
                    data.x_s_batch,
                    data.x_n_batch,
                    data.num_graphs,
                )
                # margin loss
                gamma = 0.3
                loss = F.relu(dist_p - dist_n + gamma).mean()
                loss_epoch += loss.item()
                # compute loss & backprop
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

            # Compute accuracy and log to TensorBoard
            accuracy = self.evaluate(val_loader)
            print(
                f"Epoch: {epoch}, Loss: {loss_epoch / float(len(train_loader))}, Accuracy: {accuracy}"
            )

    def evaluate(self, val_loader):
        """Evaluate the model"""
        self.model.eval()
        correct = 0
        total = 0
        with torch.no_grad():
            for data in val_loader:
                dist_p = self.model(
                    data.x_s,
                    data.x_p,
                    data.edge_index_s,
                    data.edge_index_p,
                    data.x_s_batch,
                    data.x_p_batch,
                    data.num_graphs,
                )
                dist_n = self.model(
                    data.x_s,
                    data.x_n,
                    data.edge_index_s,
                    data.edge_index_n,
                    data.x_s_batch,
                    data.x_n_batch,
                    data.num_graphs,
                )
                # margin loss
                correct += torch.where(dist_p < dist_n, 1, 0).sum().item()
                total += dist_p.size(0)
        accuracy = correct / total
        return accuracy

    def predict(self, x_s, x_t, edge_index_s, edge_index_t, x_s_batch, x_t_batch, num_graphs):
        """Predict the similarity score"""
        self.model.eval()
        with torch.no_grad():
            dist = self.model(
                x_s, x_t, edge_index_s, edge_index_t, x_s_batch, x_t_batch, num_graphs
            )
        return dist

    def save(self, name: str = "model"):
        """Save the model"""
        if self.model.state_dict():
            torch.save(
                self.model.state_dict(),
                os.path.join(self.root_abs_path, self.save_dir, f"{name}.pt"),
            )
            print(f"Save model to {os.path.join(self.root_abs_path, self.save_dir, f'{name}.pt')}")

    def load(self, name: str = "model"):
        """Load the model"""
        if self.model.state_dict():
            model_file = os.path.join(self.root_abs_path, self.save_dir, f"{name}.pt")
            if os.path.isfile(model_file):
                self.model.load_state_dict(torch.load(model_file))
                print(f"Load model from {model_file}")
            else:
                print(f"Model file {model_file} not found")

    def set_kernel(self, kernel_method: str):
        """Set the kernel method"""
        if kernel_method == "jaccard":
            self.model.kernel_method = "jaccard"
            self.model.sim_func = self.model.jaccard
        elif kernel_method == "szymkiewicz_simpson":
            self.model.kernel_method = "szymkiewicz_simpson"
            self.model.sim_func = self.model.szymkiewicz_simpson
        elif kernel_method == "head":
            self.model.kernel_method = "head"
            self.model.sim_func = self.model.head
        else:
            raise ValueError(f"Kernel method {kernel_method} not supported")


def main():
    """Main function"""
    # initialize a hydra config instance
    root_abs_path = os.path.join(os.path.dirname(__file__), "../..")
    data_abs_path = "/home/robot-learning/Data/exp_01"
    config_path = os.path.join("../..", "config/ml")
    with hydra.initialize(config_path=config_path):
        cfg = hydra.compose(config_name="gnn_matcher")

    # create data loader
    dataset = GraphTripletDataset(root=data_abs_path)
    train_dataset, val_dataset = random_split(
        dataset, [int(len(dataset) * 0.8), len(dataset) - int(len(dataset) * 0.8)]
    )
    train_loader = DataLoader(
        train_dataset, batch_size=cfg.batch_size, follow_batch=["x_s", "x_p", "x_n"], shuffle=True
    )
    val_loader = DataLoader(
        val_dataset, batch_size=cfg.batch_size, follow_batch=["x_s", "x_p", "x_n"], shuffle=True
    )

    # create model
    gnn_args = GMatchArgs(cfg)
    model = GraphSimKernelModel(threshold=0.3).to(cfg.device)
    # build matcher
    matcher = GraphMatcher(cfg, root_abs_path, model)
    matcher.set_kernel("szymkiewicz_simpson")
    # test save
    matcher.save()
    # train
    train_accuracy = matcher.evaluate(val_loader=train_loader)
    val_accuracy = matcher.evaluate(val_loader=val_loader)
    print(f"Train accuracy: {train_accuracy}, Val accuracy: {val_accuracy}")


if __name__ == "__main__":
    main()
