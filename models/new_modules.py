import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import RobertaTokenizer, RobertaModel


##################### General Modules #####################


class MLP(nn.Module):
    def __init__(
        self, input_size, hidden_size, num_hidden, output_size, activation=nn.GELU()
    ):
        super(MLP, self).__init__()
        self.layers = nn.ModuleList(
            (
                [nn.Linear(input_size, hidden_size)]
                + [nn.Linear(hidden_size, hidden_size) for _ in range(num_hidden - 1)]
                + [nn.Linear(hidden_size, output_size)]
            )
        )
        self.activation = activation

    def forward(self, x):
        for layer in self.layers[:-1]:
            x = self.activation(layer(x))
        x = self.layers[-1](x)
        return x


class XYEncoder(nn.Module):
    def __init__(self, config):
        super().__init__()

        self.mlp = MLP(
            input_size=config.input_dim + config.output_dim,
            hidden_size=config.xy_transf_dim,
            num_hidden=config.xy_encoder_num_hidden,
            output_size=config.xy_transf_dim,
        )

    def forward(self, x, y):
        xy = torch.cat([x, y], dim=-1)
        return self.mlp(xy)


##################### Dataset Encoder Modules #####################


class MAB(nn.Module):
    def __init__(self, dim_Q, dim_K, dim_V, num_heads, activation=nn.ReLU(), ln=False):
        super(MAB, self).__init__()
        self.dim_V = torch.tensor(dim_V, requires_grad=False)
        self.num_heads = num_heads
        self.fc_q = nn.Linear(dim_Q, dim_V)
        self.fc_k = nn.Linear(dim_K, dim_V)
        self.fc_v = nn.Linear(dim_K, dim_V)
        if ln:
            self.ln0 = nn.LayerNorm(dim_V)
            self.ln1 = nn.LayerNorm(dim_V)
        self.fc_o = nn.Linear(dim_V, dim_V)
        self.activation = activation

    def forward(self, Q, K):
        Q = self.fc_q(Q)
        K = self.fc_k(K)
        V = self.fc_v(K)

        dim_split = self.dim_V // self.num_heads
        Q_ = torch.cat(Q.split(dim_split, 2), 0)
        K_ = torch.cat(K.split(dim_split, 2), 0)
        V_ = torch.cat(V.split(dim_split, 2), 0)

        A = torch.softmax(Q_.bmm(K_.transpose(1, 2)) / torch.sqrt(self.dim_V), 2)
        O = torch.cat((Q_ + A.bmm(V_)).split(Q.size(0), 0), 2)
        O = O if getattr(self, "ln0", None) is None else self.ln0(O)
        O = O + self.activation(self.fc_o(O))
        O = O if getattr(self, "ln1", None) is None else self.ln1(O)
        return O


class SAB(nn.Module):
    def __init__(self, input_size, output_size, num_heads, ln=False):
        super(SAB, self).__init__()
        self.mab = MAB(input_size, input_size, output_size, num_heads, ln=ln)

    def forward(self, x):
        return self.mab(x, x)


class ISAB(nn.Module):
    def __init__(self, input_size, output_size, num_heads, num_inds, ln=False):
        super(ISAB, self).__init__()
        self.I = nn.Parameter(torch.Tensor(1, num_inds, output_size))
        nn.init.xavier_uniform_(self.I)
        self.mab0 = MAB(output_size, input_size, output_size, num_heads, ln=ln)
        self.mab1 = MAB(input_size, output_size, output_size, num_heads, ln=ln)

    def forward(self, x):
        q = self.I.repeat(x.size(0), 1, 1)
        H = self.mab0(q, x)
        return self.mab1(x, H)


class PMA(nn.Module):
    def __init__(self, dim, num_heads, num_seeds, ln=False):
        super(PMA, self).__init__()
        self.S = nn.Parameter(torch.Tensor(1, num_seeds, dim))
        nn.init.xavier_uniform_(self.S)
        self.mab = MAB(dim, dim, dim, num_heads, ln=ln)

    def forward(self, x):
        output = self.mab(self.S.repeat(x.size(0), 1, 1), x)
        return output


class SetTransformer(nn.Module):
    def __init__(self, config):
        super(SetTransformer, self).__init__()
        self.enc = nn.Sequential(
            ISAB(
                input_size=config.xy_transf_dim,
                output_size=config.xy_transf_dim,
                num_heads=config.set_transformer_num_heads,
                num_inds=config.set_transformer_num_inds,
                ln=config.set_transformer_ln,
            ),
            ISAB(
                input_size=config.xy_transf_dim,
                output_size=config.xy_transf_dim,
                num_heads=config.set_transformer_num_heads,
                num_inds=config.set_transformer_num_inds,
                ln=config.set_transformer_ln,
            ),
        )
        self.dec = nn.Sequential(
            PMA(
                dim=config.xy_transf_dim,
                num_heads=config.set_transformer_num_heads,
                num_seeds=config.set_transformer_num_seeds,
                ln=config.set_transformer_ln,
            ),
            SAB(
                input_size=config.xy_transf_dim,
                output_size=config.xy_transf_dim,
                num_heads=config.set_transformer_num_heads,
                ln=config.set_transformer_ln,
            ),
            SAB(
                input_size=config.xy_transf_dim,
                output_size=config.xy_transf_dim,
                num_heads=config.set_transformer_num_heads,
                ln=config.set_transformer_ln,
            ),
            nn.Linear(config.xy_transf_dim, config.dataset_representation_dim),
        )

    def forward(self, x):
        return self.dec(self.enc(x))


class DatasetEncoder(nn.Module):
    def __init__(self, config):
        super().__init__()

        self.config = config

        if config.dataset_encoder_type == "set_transformer":
            self.encoder = SetTransformer(config)
        else:
            raise NotImplementedError

    def forward(self, xy_context_encoded):
        if self.config.dataset_encoder_type == "set_transformer":
            t = self.encoder(xy_context_encoded)
        return t


##################### Knowledge Encoder Modules #####################


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


##################### Understanding Encoder Modules #####################


class UnderstandingEncoder(nn.Module):
    def __init__(self, config):
        super().__init__()

        if config.knowledge_dataset_merge == "sum":
            assert (
                config.dataset_representation_dim == config.knowledge_representation_dim
            )
            input_dim = config.dataset_representation_dim
        elif config.knowledge_dataset_merge == "concat":
            input_dim = (
                config.dataset_representation_dim + config.knowledge_representation_dim
            )
        elif config.knowledge_dataset_merge == "mlp":
            input_dim = config.understanding_representation_dim
            self.knowledge_dataset_merger = MLP(
                input_size=config.dataset_representation_dim
                + config.knowledge_representation_dim,
                hidden_size=config.understanding_representation_dim,
                num_hidden=config.knowledge_dataset_merger_num_hidden,
                output_size=config.understanding_representation_dim,
            )
        else:
            raise NotImplementedError

        if config.understanding_encoder_num_hidden > 0:
            self.understanding_encoder = MLP(
                input_size=input_dim,
                hidden_size=config.understanding_representation_dim,
                num_hidden=config.understanding_encoder_num_hidden,
                output_size=config.understanding_representation_dim,
            )
        else:
            self.understanding_encoder = nn.Linear(
                input_dim, config.understanding_representation_dim
            )

        self.config = config

    def forward(self, t, k, knowledge_available):
        if self.config.knowledge_dataset_merge == "sum":
            encoder_input = F.relu(t + k)
        elif self.config.knowledge_dataset_merge == "concat":
            encoder_input = torch.cat([t, k], dim=-1)
        elif self.config.knowledge_dataset_merge == "mlp":
            if knowledge_available:
                encoder_input = self.knowledge_dataset_merger(torch.cat([t, k], dim=-1))
            else:
                encoder_input = F.relu(t)

        u = self.understanding_encoder(encoder_input)

        return u


##################### Context-Target Data Interaction Modules #####################


class DataInteractionEncoder(nn.Module):
    def __init__(self, config):
        super().__init__()

        self.xy_mlp = MLP(
            input_size=config.input_dim + config.output_dim,
            hidden_size=config.data_interaction_dim,
            num_hidden=config.data_interaction_mlp_num_hidden,
            output_size=config.data_interaction_dim,
        )
        self.x_mlp = MLP(
            input_size=config.input_dim,
            hidden_size=config.data_interaction_dim,
            num_hidden=config.data_interaction_mlp_num_hidden,
            output_size=config.data_interaction_dim,
        )

        self.self_attention = nn.MultiheadAttention(
            embed_dim=config.data_interaction_self_attention_hidden_dim,
            num_heads=config.data_interaction_self_attention_num_heads,
            batch_first=True,
        )
        self.cross_attention = nn.MultiheadAttention(
            embed_dim=config.data_interaction_cross_attention_hidden_dim,
            num_heads=config.data_interaction_cross_attention_num_heads,
            batch_first=True,
        )

    def forward(self, x_context, y_context, x_target):
        xy_context_encoded = torch.cat([x_context, y_context], dim=-1)
        xy_context_encoded = self.xy_mlp(xy_context_encoded)

        x_target_encoded = self.x_mlp(x_target)

        rs = self.self_attention(
            xy_context_encoded, xy_context_encoded, xy_context_encoded
        )[0]
        rs = self.cross_attention(x_target_encoded, xy_context_encoded, rs)[0]

        r = torch.mean(rs, dim=1, keepdim=True)

        return r


##################### Memory Module #####################


class Memory(nn.Module):
    def __init__(self, config):
        super().__init__()

        self.memory_knowledge = torch.empty(
            config.knowledge_representation_dim,
            config.memory_slots,
            requires_grad=False,
        )
        self.memory_understanding = torch.empty(
            config.understanding_representation_dim,
            config.memory_slots,
            requires_grad=False,
        )

        nn.init.orthogonal_(self.memory_knowledge)
        nn.init.orthogonal_(self.memory_understanding)

        self.register_buffer(
            "w_w", torch.zeros((config.memory_slots,), requires_grad=False)
        )
        self.register_buffer(
            "w_u", torch.zeros((config.memory_slots,), requires_grad=False)
        )
        self.register_buffer(
            "w_lu", torch.zeros((config.memory_slots,), requires_grad=False)
        )

        self.beta_param = nn.Parameter(torch.tensor(0.0), requires_grad=True)

        self.config = config

    def forward(self, k, u):
        u_refined = torch.zeros(
            (u.shape[0], 1, self.config.understanding_representation_dim)
        ).to(self.config.device)

        # Perform memory read and write iteratively for each task
        for i, (k_task, u_task) in enumerate(zip(k, u)):
            # STEP 1: Calculate w_r based on cosine similarity between k and memory knowledge
            knowledge_similarity = torch.matmul(k_task, self.memory_knowledge) / (
                torch.norm(k_task, dim=-1, keepdim=True)
                * torch.norm(self.memory_knowledge, dim=0, keepdim=True)
            )  # [1, k_dim] @ [k_dim, m_slots] -> [1, m_slots]
            w_r = torch.softmax(knowledge_similarity, dim=-1).squeeze()  # [m_slots]

            # STEP 2: Erase the least used understanding slot entry
            min_idx = torch.argmin(self.w_u)

            self.memory_knowledge[:, min_idx] = 0
            self.memory_understanding[:, min_idx] = 0

            # STEP 3: Calculate the updated w_u
            self.w_u = self.config.memory_gamma * self.w_u + w_r + self.w_w

            # STEP 4: Calculate the updated w_w
            beta = torch.sigmoid(self.beta_param)
            self.w_w = beta * w_r.mean(dim=0).squeeze() + (1 - beta) * self.w_lu

            # STEP 5: Calculate the updated w_lu
            min_w_u_idx = torch.argmin(self.w_u)

            self.w_lu = torch.zeros_like(self.w_u)
            self.w_lu[min_w_u_idx] = 1

            # STEP 6: Update understanding entries in memory
            self.memory_understanding = self.memory_understanding + torch.matmul(
                u_task.T, self.w_w.unsqueeze(0)
            )

            # STEP 7: Obtain refined understanding value
            u_task_refined = (
                torch.matmul(w_r, self.memory_knowledge.t()).unsqueeze(0).unsqueeze(0)
            )

            u_refined[i] = u_task_refined

            # STEP 8: Update knowledge entries in memory
            self.memory_knowledge = self.memory_knowledge + torch.matmul(
                k_task.T, self.w_w.unsqueeze(0)
            )

        return u_refined


##################### Latent Encoder Module #####################


class LatentEncoder(nn.Module):
    def __init__(self, config):
        super().__init__()

        if config.data_interaction_understanding_merge == "sum":
            assert (
                config.data_interaction_dim == config.understanding_representation_dim
            )
            input_dim = config.understanding_representation_dim
        elif config.data_interaction_understanding_merge == "concat":
            input_dim = (
                config.data_interaction_dim + config.understanding_representation_dim
            )
        elif config.data_interaction_understanding_merge == "mlp":
            input_dim = config.data_interaction_dim
            self.data_interaction_understanding_merger = MLP(
                input_size=config.data_interaction_dim
                + config.understanding_representation_dim,
                hidden_size=config.data_interaction_understanding_merger_hidden_dim,
                num_hidden=config.data_interaction_understanding_merger_num_hidden,
                output_size=config.data_interaction_dim,
            )
        else:
            raise NotImplementedError

        if config.latent_encoder_num_hidden > 0:
            self.latent_encoder = MLP(
                input_size=input_dim,
                hidden_size=config.latent_encoder_hidden_dim,
                num_hidden=config.latent_encoder_num_hidden,
                output_size=2 * config.latent_encoder_hidden_dim,
            )
        else:
            self.latent_encoder = nn.Linear(
                input_dim, 2 * config.latent_encoder_hidden_dim
            )

        self.config = config

    def forward(self, r, u):
        if self.config.data_interaction_understanding_merge == "sum":
            encoder_input = F.relu(r + u)
        elif self.config.data_interaction_understanding_merge == "concat":
            encoder_input = torch.cat([r, u], dim=-1)
        elif self.config.data_interaction_understanding_merge == "mlp":
            encoder_input = self.data_interaction_understanding_merger(
                torch.cat([r, u], dim=-1)
            )

        q_z_stats = self.latent_encoder(encoder_input)

        return q_z_stats


##################### Decoder Module #####################


class Decoder(nn.Module):
    def __init__(self, config):
        super().__init__()

        if config.decoder_activation == "relu":
            activation = nn.ReLU()
        else:
            activation = nn.GELU()
        self.mlp = MLP(
            input_size=config.latent_encoder_hidden_dim + config.input_dim,
            hidden_size=config.decoder_hidden_dim,
            num_hidden=config.decoder_num_hidden,
            output_size=2 * config.output_dim,
            activation=activation,
        )

    def forward(self, x_target, u_target):
        """
        Decode the target set given the target dependent representation

        u_target [num_samples, bs, num_targets, understanding_representation_dim]
        x_target [bs, num_targets, input_dim]
        """
        x_target = x_target.unsqueeze(0).expand(u_target.shape[0], -1, -1, -1)
        xu_target = torch.cat([x_target, u_target], dim=-1)
        p_y_stats = self.mlp(xu_target)

        return p_y_stats
