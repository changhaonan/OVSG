from __future__ import annotations
from transformers import AutoTokenizer, AutoModel
import torch
import torch.nn.functional as F
from ovsg.env.algo.notion import Space, Domain


def mean_pooling(model_output, attention_mask):
    """Mean Pooling - Take attention mask into account for correct averaging"""
    token_embeddings = model_output[
        0
    ]  # First element of model_output contains all token embeddings
    input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
    return torch.sum(token_embeddings * input_mask_expanded, 1) / torch.clamp(
        input_mask_expanded.sum(1), min=1e-9
    )


class SentenceTransformer:
    def __init__(self, device) -> None:
        self.device = device
        # Load model from HuggingFace Hub
        self.tokenizer = AutoTokenizer.from_pretrained("sentence-transformers/all-MiniLM-L6-v2")
        self.model = AutoModel.from_pretrained("sentence-transformers/all-MiniLM-L6-v2").to(device)

    def __call__(self, sentences: str):
        # Tokenize sentences
        encoded_input = self.tokenizer(
            sentences, padding=True, truncation=True, return_tensors="pt"
        ).to(self.device)
        # Compute token embeddings
        with torch.no_grad():
            model_output = self.model(**encoded_input)

        # Perform pooling
        sentence_embeddings = mean_pooling(model_output, encoded_input["attention_mask"])

        # Normalize embeddings
        sentence_embeddings = F.normalize(sentence_embeddings, p=2, dim=1)
        return sentence_embeddings


def parse_space_domain(type_str: str) -> tuple[Space, Domain]:
    """Parse space and domain from type string"""
    if type_str.lower() == "region":
        domain = Domain.RGN
        space = Space.REGION
    elif type_str.lower() == "user":
        domain = Domain.USR
        space = Space.DYNAMIC
    elif type_str.lower() == "object":
        domain = Domain.INS
        space = Space.STATIC
    else:
        raise ValueError(f"Unknown target type: {type_str}")
    return space, domain
