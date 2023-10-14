"""Prompt manager"""
import os
import abc
from abc import ABC
import glob
import xml.etree.ElementTree as ET
import pickle

# compute similarity using sequence_transformer
from transformers import AutoTokenizer, AutoModel
import torch
import torch.nn.functional as F


def mean_pooling(model_output, attention_mask):
    """Mean Pooling - Take attention mask into account for correct averaging"""
    token_embeddings = model_output[0]  # First element of model_output contains all token embeddings
    input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
    return torch.sum(token_embeddings * input_mask_expanded, 1) / torch.clamp(input_mask_expanded.sum(1), min=1e-9)


class SentenceTransformer:
    def __init__(self, device) -> None:
        self.device = device
        # Load model from HuggingFace Hub
        self.tokenizer = AutoTokenizer.from_pretrained("sentence-transformers/all-MiniLM-L6-v2")
        self.model = AutoModel.from_pretrained("sentence-transformers/all-MiniLM-L6-v2").to(device)

    def __call__(self, sentences: str):
        # Tokenize sentences
        encoded_input = self.tokenizer(sentences, padding=True, truncation=True, return_tensors="pt").to(self.device)
        # Compute token embeddings
        with torch.no_grad():
            model_output = self.model(**encoded_input)

        # Perform pooling
        sentence_embeddings = mean_pooling(model_output, encoded_input["attention_mask"])

        # Normalize embeddings
        sentence_embeddings = F.normalize(sentence_embeddings, p=2, dim=1)
        return sentence_embeddings


class PromptManager(ABC):
    # Set this in SOME subclasses
    metadata = {}

    # Set this in ALL subclasses
    name = ""

    @abc.abstractmethod
    def get_prompt(self, task_type, query, query_mode, **kwargs):
        """Return prompt for task"""
        raise NotImplementedError


class PromptManagerDataBase(PromptManager):
    def __init__(self, cfg):
        self.db_dir = os.path.join(cfg.prompt_dir, "db")
        self.sentense_transformer = SentenceTransformer(cfg.prompt_device)
        self.top_k = cfg.prompt_top_k

    def get_prompt(self, task_type, query="", query_mode="bf", **kwargs):
        """Return prompt for task"""
        if query_mode == "bf":
            return self.query_prompt_bf(task_type)
        elif query_mode == "none":
            return ""
        elif query_mode == "sim":
            kwargs["redo"] = True  # Force redo now
            return self.query_promt_by_sim(task_type, query, **kwargs)
        else:
            raise ValueError(f"{query_mode} mode is not supported.")

    def query_prompt_bf(self, task_type):
        """Query prompt from database by brute force"""
        prompt_list = []
        for file in glob.glob(self.db_dir + f"/exp_{task_type}*.xml"):
            with open(file, "r") as f:
                xml_string = f.read()
            root = ET.fromstring(xml_string)
            round = root.findall("round")
            for r in round:
                user = r.find("user")
                if user is not None:
                    prompt_list.append({"user": user.text})
                assistant = r.find("assistant")
                if assistant is not None:
                    prompt_list.append({"assistant": assistant.text})
        return prompt_list

    def query_promt_by_sim(self, task_type, query: str, **kwargs):
        """Query prompt from database by similarity"""
        # parse kwargs
        redo = kwargs.get("redo", False)
        # compute query embedding
        query_embedding = self.sentense_transformer(query)
        # load prompt database
        prompt_tag_dict = self.load_prompt_db(task_type, redo)
        def similarity_func(x): return float(F.cosine_similarity(query_embedding, self.sentense_transformer(x[0])))
        sorted_prompt_tag_dict = sorted(prompt_tag_dict.items(), key=similarity_func, reverse=True)
        # return top k prompt
        top_k = self.top_k if len(sorted_prompt_tag_dict) > self.top_k else len(sorted_prompt_tag_dict)
        top_prompt_file_list = [x[1] for x in sorted_prompt_tag_dict[:top_k]]

        # verbose
        print("------------- Query prompt by similarity -------------")
        print("Query: ", query)
        print("Top similar prompts:")
        for i in range(top_k):
            print(f"Prompt {i}: {sorted_prompt_tag_dict[i][0]}")
        print("------------------------------------------------------")

        # return prompt list
        prompt_list = []
        for file in top_prompt_file_list:
            with open(file, "r") as f:
                xml_string = f.read()
            root = ET.fromstring(xml_string)
            round = root.findall("round")
            for r in round:
                user = r.find("user")
                if user is not None:
                    prompt_list.append({"user": user.text})
                assistant = r.find("assistant")
                if assistant is not None:
                    prompt_list.append({"assistant": assistant.text})
        return prompt_list

    def query_promt_by_llm(self, task_type, query: str, **kwargs):
        """Query prompt from database by a llm"""
        pass

    def build_prompt_db(self, task_type):
        """Build prompt database"""
        prompt_tag_dict = {}
        for file in glob.glob(self.db_dir + f"/exp_{task_type}_*.xml"):
            with open(file, "r") as f:
                xml_string = f.read()
            root = ET.fromstring(xml_string)
            tag = root.findall("tag")
            if tag is not None:
                for t in tag:
                    prompt_tag_dict[t.text] = file
        return prompt_tag_dict

    def save_prompt_db(self, task_type):
        """Save prompt database"""
        prompt_tag_dict = self.build_prompt_db(task_type)
        # save dict into compressed file
        with open(self.db_dir + f"/tag_{task_type}.pkl", "wb") as f:
            pickle.dump(prompt_tag_dict, f, pickle.HIGHEST_PROTOCOL)
        return prompt_tag_dict

    def load_prompt_db(self, task_type, redo=False):
        if redo or not os.path.exists(self.db_dir + f"/tag_{task_type}.pkl"):
            return self.save_prompt_db(task_type)
        else:
            """Load prompt database"""
            with open(self.db_dir + f"/tag_{task_type}.pkl", "rb") as f:
                prompt_tag_dict = pickle.load(f)
            return prompt_tag_dict
