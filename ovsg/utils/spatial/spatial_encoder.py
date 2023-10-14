""" Currently, we want to train a multi-label prediction network"""
from __future__ import annotations
import os
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader, random_split
import torch.nn as nn
import torch.nn.functional as F
import hydra
from ovsg.utils.spatial.spatial import Points9 as p9
from ovsg.utils.spatial.spatial import SpatialRelSampler


def build_txt_embedding(txt_encoder, device):
    """Build txt embedding"""
    vocab_map = p9.vocabulary_map()
    if txt_encoder == "clip":
        from clip import clip

        model, preprocess = clip.load("ViT-B/32", device=device)
        # return {k: tokensize(v) for k, v in vocab_map.items()}
        def encode(x): return model.encode_text(clip.tokenize(x).to(device)).detach().cpu().numpy()
        return {k: encode(v) for k, v in vocab_map.items()}
    else:
        raise NotImplementedError


class SpatialRelDataset(Dataset):
    """Spatial relationship dataset"""

    def __init__(self, poses, labels, txt_encoder, device, transform=None):
        self.poses = poses
        self.labels = labels
        self.transform = transform
        # param
        self.device = device
        # rel sampler
        self.rel_sampler = SpatialRelSampler(device)

    def __len__(self):
        return len(self.poses)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        pose = self.poses[idx]
        label = self.labels[idx]

        # create positive and negative spatial relationship
        pos_rel_embedding, neg_rel_embedding = self.rel_sampler.sample_rel_embedding(label)

        txt_embedding = np.vstack([pos_rel_embedding, neg_rel_embedding])
        sim = np.vstack([np.ones((1, 1)), np.zeros((1, 1))])
        pose = np.vstack([pose, pose])

        if self.transform:
            pose = self.transform(pose)

        return {
            "pose": pose,
            "txt_embedding": txt_embedding,
            "sim": sim,
        }


class SpatialRelModel(nn.Module):
    """Spatial relationship model"""

    def __init__(self, spatial_input_size, txt_input_size, hidden_size, embedding_size):
        super(SpatialRelModel, self).__init__()
        # encode spatial relationship
        self.spatial_encoder = nn.Sequential(
            nn.Linear(spatial_input_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, embedding_size),
        )
        # encode text
        self.txt_encoder_head = nn.Sequential(
            nn.Linear(txt_input_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, embedding_size),
        )
        # predict similarity
        self.predict_head = nn.Sequential(
            nn.Linear(embedding_size * 2, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, 1),
        )

    def forward(self, pose_pair, txt_embedding):
        """Forward
        Args:
            pose_pair: (batch, 18)
            txt_embedding: (batch, 512)
        Return sim: (batch, 1)
        """
        spatial_embedding = self.spatial_encoder(pose_pair)
        txt_embedding = self.txt_encoder_head(txt_embedding)
        sim = self.predict_head(torch.cat([spatial_embedding, txt_embedding], dim=1))
        return sim


def criterion(pred, sim):
    """Loss function
    Args:
        pred: (batch, 9)
    """
    # compute loss
    loss = F.binary_cross_entropy_with_logits(pred, sim)
    return loss


class SpatialRelEncoder:
    """Spatial relationship encoder"""

    def __init__(self, cfg, root_abs_path, model, train_loader=None, val_loader=None):
        self.model = model
        self.cfg = cfg
        self.train_loader = train_loader
        self.val_loader = val_loader

        # param
        self.device = cfg.device
        self.lr_rate = cfg.lr_rate
        self.epochs = cfg.epochs
        self.save_dir = cfg.save_dir
        self.root_abs_path = root_abs_path

    def train(self):
        """Train model"""
        num_epochs = self.epochs
        losses = []
        checkpoint_losses = []

        # use AdamW
        optimizer = torch.optim.AdamW(self.model.parameters(), lr=self.lr_rate)
        n_total_steps = len(self.train_loader)

        self.model.train()
        for epoch in range(num_epochs):
            for batch_idx, data in enumerate(self.train_loader):
                # load data
                pose_pair, txt_embedding, sim = (
                    data["pose"].to(self.device).to(torch.float32),
                    data["txt_embedding"].to(self.device).to(torch.float32),
                    data["sim"].to(self.device).to(torch.float32),
                )
                # reshape
                pose_pair = pose_pair.reshape(-1, 18)
                txt_embedding = txt_embedding.reshape(-1, 512)
                sim = sim.reshape(-1, 1)
                # compute loss
                pred = self.model(pose_pair, txt_embedding)
                loss = criterion(pred, sim)
                losses.append(loss.item())

                # compute gradient and do optimizer step
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

            # verbose
            if epoch % 10 == 0:
                # eval
                self.model.eval()
                with torch.no_grad():
                    val_loss = 0
                    for data in self.val_loader:
                        # load data
                        pose_pair, txt_embedding, sim = (
                            data["pose"].to(self.device).to(torch.float32),
                            data["txt_embedding"].to(self.device).to(torch.float32),
                            data["sim"].to(self.device).to(torch.float32),
                        )
                        # reshape
                        pose_pair = pose_pair.reshape(-1, 18)
                        txt_embedding = txt_embedding.reshape(-1, 512)
                        sim = sim.reshape(-1, 1)
                        # compute loss
                        pred = self.model(pose_pair, txt_embedding)
                        loss = criterion(pred, sim)
                        val_loss += loss
                    val_loss /= len(self.val_loader.dataset)
                    print(
                        f"{epoch}: Val set: Average loss: {val_loss:.4f}",
                    )

        return checkpoint_losses

    def encode_spatial(self, pose_pair: np.ndarray | torch.Tensor):
        """Encode pose"""
        if isinstance(pose_pair, np.ndarray):
            pose_pair = torch.from_numpy(pose_pair).to(self.device).to(torch.float32)

        self.model.eval()
        with torch.no_grad():
            spatial_embedding = self.model.spatial_encoder(pose_pair)
            return spatial_embedding

    def encode_text(self, txt_embedding: np.ndarray | torch.Tensor):
        """Encode txt"""
        if isinstance(txt_embedding, np.ndarray):
            txt_embedding = torch.from_numpy(txt_embedding).to(self.device).to(torch.float32)

        self.model.eval()
        with torch.no_grad():
            txt_embedding = self.model.txt_encoder_head(txt_embedding)
            return txt_embedding

    def predict_from_pose(
        self,
        pose_pair: np.ndarray | torch.Tensor,
        txt_embedding: np.ndarray | torch.Tensor,
    ):
        """Predict similarity"""
        # type check
        if isinstance(pose_pair, np.ndarray):
            pose_pair = torch.from_numpy(pose_pair).to(self.device).to(torch.float32)
        if isinstance(txt_embedding, np.ndarray):
            txt_embedding = torch.from_numpy(txt_embedding).to(self.device).to(torch.float32)
        # shape check
        if len(pose_pair.shape) == 1:
            pose_pair = pose_pair.reshape(1, -1)
        if len(txt_embedding.shape) == 1:
            txt_embedding = txt_embedding.reshape(1, -1)

        self.model.eval()
        with torch.no_grad():
            spatial_embedding = self.model.spatial_encoder(pose_pair)
            txt_embedding = self.model.txt_encoder_head(txt_embedding)
            sim = self.model.predict_head(torch.cat([spatial_embedding, txt_embedding], dim=1))
            # sigmoid
            sim = torch.sigmoid(sim)
            return sim

    def predict_from_embedding(
        self,
        spatial_embedding: np.ndarray | torch.Tensor,
        txt_embedding: np.ndarray | torch.Tensor,
    ):
        """Predict similarity"""
        # type check
        if isinstance(spatial_embedding, np.ndarray):
            spatial_embedding = (
                torch.from_numpy(spatial_embedding).to(self.device).to(torch.float32)
            )
        if isinstance(txt_embedding, np.ndarray):
            txt_embedding = torch.from_numpy(txt_embedding).to(self.device).to(torch.float32)
        # shape check
        if len(spatial_embedding.shape) == 1:
            spatial_embedding = spatial_embedding.reshape(1, -1)
        if len(txt_embedding.shape) == 1:
            txt_embedding = txt_embedding.reshape(1, -1)

        self.model.eval()
        with torch.no_grad():
            # apply txt encoder head
            txt_embedding = self.model.txt_encoder_head(txt_embedding)
            sim = self.model.predict_head(torch.cat([spatial_embedding, txt_embedding], dim=1))
            # sigmoid
            sim = torch.sigmoid(sim)
            return sim

    def predict(self, pose_pair, txt_embedding_dict):
        """Predict"""
        pose_embedding = self.encode_spatial(pose_pair).detach().cpu().numpy()
        # compute simiarity with all txt
        pose_txt_list = []
        description_seen = []  # avoid duplicate
        for key, txt_embedding in txt_embedding_dict.items():
            max_sim = -np.inf
            for i in range(txt_embedding.shape[0]):
                sim = (
                    self.predict_from_pose(pose_pair, txt_embedding[i].reshape(1, -1))
                    .detach()
                    .cpu()
                    .numpy()[0]
                )
                if sim > max_sim:
                    max_sim = sim
            description = p9.vocabulary(key)
            if description not in description_seen:
                description_seen.append(description)
                pose_txt_list.append((description, max_sim))
        # sort
        pose_txt_list.sort(key=lambda x: x[1], reverse=True)
        return pose_txt_list

    # model save and load
    def save(self):
        """Save model"""
        torch.save(
            self.model.state_dict(),
            os.path.join(
                os.path.join(self.root_abs_path, self.save_dir),
                "model.pth",
            ),
        )
        print(f"Spatial encoder is saved to {self.save_dir}")

    def load(self):
        """Load model"""
        self.model.load_state_dict(
            torch.load(
                os.path.join(
                    os.path.join(self.root_abs_path, self.save_dir),
                    "model.pth",
                )
            )
        )
        print(f"Spatial encoder loaded from {self.save_dir}")


if __name__ == "__main__":
    # initialize a hydra config instance
    root_abs_path = os.path.join(os.path.dirname(__file__), "../..")
    config_path = os.path.join("../..", "config/ml")
    with hydra.initialize(config_path=config_path):
        cfg = hydra.compose(config_name="spatial_encoder")

    # generate random data
    debug = False
    if not debug:
        poses_a, poses_b = p9.random_pose_pair(num_sample=cfg.num_data)
        labels = np.vstack([p9.label(pose_a, pose_b) for pose_a, pose_b in zip(poses_a, poses_b)])
        pose_data = np.concatenate((poses_a, poses_b), axis=1)

        # create data loader
        val_size = int(0.2 * len(pose_data))
        train_dataset, val_dataset = random_split(
            SpatialRelDataset(
                poses=pose_data, labels=labels, txt_encoder=cfg.txt_encoder, device=cfg.device
            ),
            [len(pose_data) - val_size, val_size],
            generator=torch.Generator().manual_seed(42),
        )
        train_loader = DataLoader(train_dataset, batch_size=cfg.batch_size, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=cfg.batch_size, shuffle=True)
    else:
        train_loader = None
        val_loader = None

    # create model
    model = SpatialRelModel(
        spatial_input_size=cfg.spatial_input_size,
        txt_input_size=cfg.txt_input_size,
        hidden_size=cfg.hidden_size,
        embedding_size=cfg.embedding_size,
    ).to(cfg.device)

    # build trainer
    trainer = SpatialRelEncoder(
        cfg,
        root_abs_path,
        model,
        train_loader,
        val_loader,
    )

    if not debug:
        trainer.train()
        trainer.save()
    else:
        # test
        trainer.load()
    # trainer.test()
    vocab_embedding_map = build_txt_embedding(txt_encoder=cfg.txt_encoder, device=cfg.device)

    # load from test data
    pose_data = np.load(os.path.join(root_abs_path, "data/notion/instance_poses.npy"))
    pose_a = pose_data[:, :9]
    pose_b = pose_data[:, 9:]
    normalized_pose_pair = p9.normalize_pairs(pose_a, pose_b)

    for i in range(10):
        pose_pair = normalized_pose_pair[i].reshape(1, -1)
        print(f"pose pair {pose_pair}:")
        p9.visualize(pose_pair[0])
        pose_txt_list = trainer.predict(pose_pair, vocab_embedding_map)
        gt_label = p9.label(pose_pair[0, :9], pose_pair[0, 9:])
        # compute similarity with gt label
        spatial_embedding = trainer.encode_spatial(pose_pair)
        txt_embedding = vocab_embedding_map[tuple(gt_label)][0]
        gt_sim = trainer.predict_from_embedding(spatial_embedding, txt_embedding)
        print(f"gt: {p9.vocabulary(label=gt_label)}, sim: {gt_sim}")
        for j in range(10):  # top 10
            print(f"{j}:{pose_txt_list[j][0]}, sim: {pose_txt_list[j][1]}")
        print("------------------")
