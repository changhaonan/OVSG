from __future__ import annotations
import gensim.downloader as api
import numpy as np
import clip
from PIL import Image
from ovsg.env.algo.notion import (
    Notion,
    NotionTxt,
    NotionImg,
    NotionUser,
    NotionGraph,
    NotionLink,
    Space,
    User,
    Domain,
    Feature,
    FeatureType,
    NotionConstants,
)
from ovsg.env.algo.notion_utils import SentenceTransformer


class NotionGraphEncoder(NotionGraph):
    """NotionGraph using Encoder"""

    def __init__(self, cfg, **kwargs):
        super().__init__()
        # encoder config
        self.txt_encoder = cfg.notion_txt_encoder
        self.usr_encoder = cfg.notion_usr_encoder
        self.rel_encoder = cfg.notion_rel_encoder
        self.img_encoder = cfg.notion_img_encoder
        self.ins_encoder = cfg.notion_ins_encoder
        self.encoders = list(
            set(self.txt_encoder + self.rel_encoder + self.img_encoder + self.ins_encoder)
        )
        # encoders that encode query
        self.query_encoders = list(set(self.txt_encoder + self.rel_encoder + self.usr_encoder))
        # init
        self.device = cfg.notion_device
        self.relations = []
        self.relations_keys = {}
        self.clip_model = None
        self.clip_preprocess = None
        self.sentence_transformer = None
        self.word2vec = None

    def init_model(self):
        """Lazy load model"""
        # init clip
        if "clip" in self.encoders:
            print("Loading CLIP ...")
            self.clip_model, self.clip_preprocess = clip.load("ViT-B/32", self.device)
        # init sentence transformer
        if "st" in self.encoders:
            print("Loading Sentence Transformer ...")
            self.sentence_transformer = SentenceTransformer(self.device)
        # init word2vec
        if "wv" in self.encoders:
            print("Loading Word2Vec ...")
            self.word2vec = api.load("glove-wiki-gigaword-300")

    def add_notion(
        self,
        keys: list[Feature],
        notion: Notion | str | np.ndarray | Image.Image | User | None,
        address: np.ndarray,
        domain: Domain,
        name: str,
        space: Space,
        **kwargs,
    ) -> int:
        """Add notion to NotionGraph"""
        if isinstance(notion, Notion):
            return super().add_notion(keys, notion, address, domain, name, space, **kwargs)
        else:
            notion = self._build_notion(keys, notion, address, domain, name, space, **kwargs)
            return super().add_notion([], notion, address, domain, name, space, **kwargs)

    def link(
        self,
        notion1: Notion | int,
        relation: str,
        notion2: Notion | int,
        confidence: float = 1.0,
        relation_keys: list[Feature] | None = None,
    ):
        """Link two notions"""
        if isinstance(notion1, int):
            notion1 = self.notions[notion1]
        if isinstance(notion2, int):
            notion2 = self.notions[notion2]
        # prepare link
        link = NotionLink(notion1.id, notion2.id, notion1.name, notion2.name)
        tuple_key = (notion1.id, notion2.id)
        if tuple_key not in self._links:
            self._links[tuple_key] = link
            notion1.forward_links.append(notion2.id)
            notion2.backward_links.append(notion1.id)
        # add relation
        relation = relation.strip()
        relation = relation.lower()

        if relation_keys is None:
            if relation not in self.relations:
                # Currently rel encoder only support one encoder
                self.relations_keys[relation] = self.encode_text(
                    text=relation, encoder=self.rel_encoder[0]
                )
                self.relations.append(relation)
            self._links[tuple_key].add_relation(self.relations_keys[relation], relation, confidence)
        else:
            # use provided relation keys
            self._links[tuple_key].add_relation(relation_keys, relation, confidence)

    def unlink(self, notion1: Notion, notion2: Notion):
        """Unlink two notions"""
        tuple_key = (notion1.id, notion2.id)
        if tuple_key in self._links:
            self._links.pop(tuple_key)

        notion1.forward_links.remove(notion2.id)
        notion2.backward_links.remove(notion1.id)

    def update(self, **kwargs):
        """Build NotionGraph"""
        # print("Building NotionGraph...")
        pass

    def _build_notion(
        self,
        key: list[np.ndarray],
        content: any,
        address: np.ndarray,
        domain: Domain,
        name: str,
        space: Space,
        **kwargs,
    ) -> Notion:
        """Build notion from content"""
        if key:
            return super()._build_notion(key, content, address, domain, name, space, **kwargs)
        elif isinstance(content, str):
            if content.strip().upper() in NotionConstants["Keywords"]:
                # If content is a keyword, use a zero vector as key
                force_zero = True
            else:
                force_zero = False
            keys = [
                self.encode_text(text=content, encoder=txt_encoder, force_zero=force_zero)
                for txt_encoder in self.txt_encoder
            ]
            notion = NotionTxt(
                address=address,
                domain=domain,
                keys=keys,
                content=content,
                forward_links=[],
                backward_links=[],
                name=name,
                space=space,
                **kwargs,
            )
            return notion
        elif isinstance(content, np.ndarray) or isinstance(content, Image.Image):
            # prepare notion
            if isinstance(content, np.ndarray):
                content = Image.fromarray(content)
                content.convert("RGB")
            keys = [
                self.encode_image(image=content, encoder=img_encoder)
                for img_encoder in self.img_encoder
            ]
            notion = NotionImg(
                address=address,
                domain=domain,
                keys=keys,
                content=content,
                forward_links=[],
                backward_links=[],
                name=name,
                space=space,
                **kwargs,
            )
            return notion
        elif isinstance(content, User):
            # prepare notion
            keys = []
            for name_str in content.names:
                for usr_encoder in self.usr_encoder:
                    keys.append(self.encode_text(text=name_str, encoder=usr_encoder))
            notion = NotionUser(
                address=address,
                domain=domain,
                keys=keys,
                content=content,
                forward_links=[],
                backward_links=[],
                name=name,
                space=space,
                **kwargs,
            )
            return notion
        else:
            raise ValueError("Invalid content")

    def search(
        self,
        query: str,
        top_k: int,
        domain: Domain | None = None,
        space: Space | None = None,
        **kwargs,
    ) -> tuple[list[Notion], list[float]]:
        """Search notions by query with clip feature"""
        if self.clip_model is None:
            self.init_model()
        # prepare query
        queries = []
        for encoder in self.query_encoders:
            queries.append(self.encode_text(query, encoder=encoder))
        # search
        return self.search_by_key(queries, top_k, domain=domain, space=space, **kwargs)

    # utilitys
    def encode_text(self, text: str, encoder: str, **kwargs) -> Feature:
        """Encode text with clip feature"""
        if self.clip_model is None:
            self.init_model()

        # parse
        force_zero = kwargs.get("force_zero", False)

        # preprocess text
        text = text.strip()
        text = text.lower()
        # text = text.replace(" ", "_")
        text = text.replace("_", " ")

        if encoder == "clip":
            clip_feature = (
                self.clip_model.encode_text(clip.tokenize(text).to(self.device))
                .cpu()
                .detach()
                .numpy()
                .flatten()
            )
            if force_zero:
                clip_feature = np.zeros(clip_feature.shape)
            return Feature(clip_feature, FeatureType.CLIPTXT)
        elif encoder == "st":
            st_feature = self.sentence_transformer(text).cpu().detach().numpy().flatten()
            if force_zero:
                st_feature = np.zeros(st_feature.shape)
            return Feature(st_feature, FeatureType.ST)
        elif encoder == "wv":
            # word2vec don't support space, use underscore instead
            text = text.replace(" ", "_")
            if text in self.word2vec.key_to_index:
                wv_feature = self.word2vec.word_vec(text)
            else:
                print(
                    Warning(f"Word {text} not in word2vec vocabulary, using zero vector instead.")
                )
                wv_feature = np.zeros(self.word2vec.vector_size)
            if force_zero:
                wv_feature = np.zeros(wv_feature.shape)
            return Feature(wv_feature, FeatureType.WV)
        else:
            raise ValueError("Invalid encoder")

    def encode_image(self, image: Image.Image, encoder: str) -> Feature:
        """Encode image with clip feature"""
        if self.clip_model is None:
            self.init_model()

        if encoder == "clip":
            image_tensor = self.clip_preprocess(image).unsqueeze(0).to(self.device)
            clip_feature = (
                self.clip_model.encode_image(image_tensor).cpu().detach().numpy().flatten()
            )
            return Feature(clip_feature, FeatureType.CLIPIMG)
        else:
            raise ValueError("Invalid encoder")
