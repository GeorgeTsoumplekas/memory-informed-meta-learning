import torch
import torch.nn as nn
from transformers import RobertaTokenizer, RobertaModel

from models.modules import MLP


class RoBERTa(nn.Module):
    def __init__(self, config):
        super(RoBERTa, self).__init__()

        self.llm = RobertaModel.from_pretrained("roberta-base")

        if config.roberta_freeze_llm:
            for name, param in self.llm.named_parameters():
                param.requires_grad = False

        if config.roberta_tune_llm_layer_norms:
            for name, param in self.llm.named_parameters():
                if "LayerNorm" in name:
                    param.requires_grad = True

        for name, param in self.llm.named_parameters():
            if name == "pooler.dense.weight" or name == "pooler.dense.bias":
                param.requires_grad = True

        self.device = config.device
        self.tokenizer = RobertaTokenizer.from_pretrained(
            "roberta-base", truncation=True, do_lower_case=True
        )

    def forward(self, knowledge):
        knowledge = self.tokenizer.batch_encode_plus(
            knowledge,
            return_tensors="pt",
            return_token_type_ids=True,
            padding=True,
            truncation=True,
        )

        input_ids = knowledge["input_ids"].to(self.device)
        attention_mask = knowledge["attention_mask"].to(self.device)
        token_type_ids = knowledge["token_type_ids"].to(self.device)

        llm_output = self.llm(
            input_ids=input_ids.squeeze(1),
            attention_mask=attention_mask.squeeze(1),
            token_type_ids=token_type_ids.squeeze(1),
        )
        hidden_state = llm_output[0]
        output = hidden_state[:, 0]

        return output


class NoEmbedding(nn.Module):
    def __init__(self, config):
        super().__init__()

        self.dim_model = config.knowledge_representation_dim
        self.device = config.device

    def forward(self, knowledge):
        # check if tensor
        if isinstance(knowledge, torch.Tensor):
            return knowledge.to(self.device)
        else:
            return torch.stack(knowledge).float().to(self.device)


class SimpleEmbedding(nn.Module):
    def __init__(self, config):
        super().__init__()

        self.dim_model = config.num_classes
        self.embedding = nn.Embedding(
            num_embeddings=self.dim_model,
            embedding_dim=self.dim_model,
        )

    def forward(self, knowledge):
        knowledge = torch.tensor(knowledge).long().to(self.embedding.weight.device)

        return self.embedding(knowledge)


class SetEmbedding(nn.Module):
    def __init__(self, config):
        super().__init__()

        self.dim_model = config.knowledge_representation_dim
        self.device = config.device
        self.h1 = MLP(
            input_size=config.knowledge_input_dim,
            hidden_size=config.knowledge_representation_dim,
            num_hidden=config.set_embedding_num_hidden,
            output_size=config.knowledge_representation_dim,
        )
        self.h2 = MLP(
            input_size=config.knowledge_representation_dim,
            hidden_size=config.knowledge_representation_dim,
            num_hidden=config.set_embedding_num_hidden,
            output_size=config.knowledge_representation_dim,
        )

    def forward(self, knowledge):
        knowledge = knowledge.to(self.device)
        ks = self.h1(knowledge)
        k = torch.sum(ks, dim=1, keepdim=True)
        k = self.h2(k)

        return k


class KnowledgeEncoder(nn.Module):
    def __init__(self, config):
        super(KnowledgeEncoder, self).__init__()

        if config.text_encoder == "roberta":
            self.text_encoder = RoBERTa(config)
        elif config.text_encoder == "simple":
            self.text_encoder = SimpleEmbedding(config)
        elif config.text_encoder == "none":
            self.text_encoder = NoEmbedding(config)
        elif config.text_encoder == "set":
            self.text_encoder = SetEmbedding(config)

        if config.knowledge_encoder_num_hidden > 0:
            self.knowledge_encoder = MLP(
                input_size=self.text_encoder.dim_model,
                hidden_size=config.knowledge_representation_dim,
                num_hidden=config.knowledge_encoder_num_hidden,
                output_size=config.knowledge_representation_dim,
            )
        else:
            self.knowledge_encoder = nn.Linear(
                self.text_encoder.dim_model, config.knowledge_representation_dim
            )

    def forward(self, knowledge):
        text_representation = self.text_encoder(knowledge)
        k = self.knowledge_encoder(text_representation)
        if k.dim() == 2:
            k = k.unsqueeze(1)

        return k
