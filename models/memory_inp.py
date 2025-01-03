import sys
import os

import torch
import torch.nn as nn
import torch.nn.functional as F

sys.path.insert(0, os.path.dirname(os.path.abspath(os.path.join(__file__, os.pardir))))
from models.new_modules import (
    DatasetEncoder,
    KnowledgeEncoder,
    UnderstandingEncoder,
    DataInteractionEncoder,
    Memory,
    LatentEncoder,
    Decoder,
    XYEncoder,
)
from models.utils import MultivariateNormalDiag


class MemoryINP(nn.Module):
    def __init__(self, config):
        super().__init__()

        self.config = config

        self.xy_encoder = XYEncoder(config)
        self.dataset_encoder = DatasetEncoder(config)
        self.knowledge_encoder = KnowledgeEncoder(config)
        self.understanding_encoder = UnderstandingEncoder(config)
        self.data_interaction_encoder = DataInteractionEncoder(config)
        self.memory = Memory(config)
        self.latent_encoder = LatentEncoder(config)
        self.decoder = Decoder(config)

        self.train_num_z_samples = config.train_num_z_samples
        self.test_num_z_samples = config.test_num_z_samples

    def forward(self, x_context, y_context, x_target, y_target, knowledge=None):
        xy_context_encoded = self.xy_encoder(x_context, y_context)

        # Get dataset representation
        t = self.dataset_encoder(xy_context_encoded)

        # Get knowledge representation
        drop_knowledge = torch.rand(1) < self.config.knowledge_dropout
        knowledge_available = (
            True if (knowledge is not None and not drop_knowledge) else False
        )

        if knowledge_available:
            k = self.knowledge_encoder(knowledge)
        else:
            k = torch.zeros(
                (t.shape[0], 1, self.config.knowledge_representation_dim)
            ).to(self.config.device)

        # Get understanding representation
        u = self.understanding_encoder(t, k, knowledge_available)

        # Get data interaction representation
        r = self.data_interaction_encoder(x_context, y_context, x_target)

        # Refine understanding representation with memory
        if knowledge_available and self.config.use_memory:
            u_refined = self.memory(k, u)
        else:
            u_refined = u

        # Get latent representation
        q_z_stats = self.latent_encoder(r, u_refined)

        # Infer distribution of z using the global representation
        q_zCc = self.infer_latent_dist(q_z_stats, self.config.latent_encoder_hidden_dim)

        # Get z samples
        z_samples, q_zCct = self.sample_latent(x_target, y_target, u_refined, q_zCc)

        # Reshape z_samples to the shape of x_target
        u_target = z_samples  # [num_z_samples, batch_size, 1, understanding_representation_dim]
        u_target = u_target.expand(-1, -1, x_target.shape[1], -1)

        # Get latent representation of y
        p_y_stats = self.decoder(x_target, u_target)

        # Infer distribution of y
        p_yCc = self.infer_latent_dist(p_y_stats, self.config.output_dim)

        return p_yCc, z_samples, q_zCc, q_zCct

    def infer_latent_dist(self, q_stats, split_dim):
        q_loc, q_scale = q_stats.split(split_dim, dim=-1)
        q_scale = 0.01 + 0.99 * F.softplus(q_scale)
        q_dist = MultivariateNormalDiag(q_loc, q_scale)

        return q_dist

    def sample_latent(self, x_target, y_target, u_refined, q_zCc):
        if y_target is not None and self.training:
            r_from_target = self.data_interaction_encoder(x_target, y_target, x_target)
            q_zt_stats = self.latent_encoder(r_from_target, u_refined)
            q_zt_loc, q_zt_scale = q_zt_stats.split(
                self.config.latent_encoder_hidden_dim, dim=-1
            )
            q_zt_scale = 0.01 + 0.99 * F.softplus(q_zt_scale)
            q_zCct = MultivariateNormalDiag(q_zt_loc, q_zt_scale)
            sampling_dist = q_zCct
        else:
            q_zCct = None
            sampling_dist = q_zCc

        if self.training:
            z_samples = sampling_dist.rsample([self.train_num_z_samples])
        else:
            z_samples = sampling_dist.rsample([self.test_num_z_samples])

        return z_samples, q_zCct


if __name__ == "__main__":
    from argparse import Namespace
    from loss import ELBOLoss
    from dataset.dataset import SetKnowledgeTrendingSinusoids
    from dataset.utils import get_dataloader
    import numpy as np
    import random

    config = Namespace(
        # model
        input_dim=1,
        output_dim=1,
        seed=44,
        dataset="set-trending-sinusoids",
        num_targets=100,
        # dataset
        batch_size=64,
        min_num_context=1,
        max_num_context=30,
        x_sampler="uniform",
        noise=0,
        device="cuda",
        # knowledge_input_dim = 128,
        dataset_encoder_type="set_transformer",
        dataset_representation_dim=128,
        set_transformer_num_heads=2,
        set_transformer_num_inds=6,
        set_transformer_ln=True,
        set_transformer_hidden_dim=128,
        set_transformer_num_seeds=1,
        xy_transf_dim=128,
        xy_encoder_num_hidden=2,
        # knowledge
        knowledge_representation_dim=128,
        text_encoder="set",
        roberta_freeze_llm=True,
        roberta_tune_llm_layer_norms=False,
        set_embedding_num_hidden=1,
        knowledge_encoder_num_hidden=2,
        knowledge_dropout=0.3,
        use_knowledge=True,
        # Understanding Encoder
        understanding_representation_dim=128,
        knowledge_dataset_merge="sum",
        knowledge_dataset_merger_hidden_dim=128,
        knowledge_dataset_merger_num_hidden=1,
        understanding_encoder_num_hidden=2,
        # Data Interaction Encoder
        data_interaction_mlp_num_hidden=2,
        data_interaction_self_attention_hidden_dim=128,
        data_interaction_self_attention_num_heads=2,
        data_interaction_cross_attention_hidden_dim=128,
        data_interaction_cross_attention_num_heads=2,
        data_interaction_dim=128,
        # Memory Module
        use_memory=True,
        memory_slots=48,
        memory_gamma=0.7,
        # Latent Encoder Module
        data_interaction_understanding_merge="sum",
        data_interaction_understanding_merger_hidden_dim=128,
        data_interaction_understanding_merger_num_hidden=2,
        latent_encoder_hidden_dim=128,
        latent_encoder_num_hidden=2,
        # Decoder Module
        decoder_activation="gelu",
        decoder_hidden_dim=64,
        decoder_num_hidden=2,
        test_num_z_samples=32,
        train_num_z_samples=1,
    )
    config.device = "cpu"

    dataset = SetKnowledgeTrendingSinusoids(split="train", knowledge_type="abc2")
    train_dataloader = get_dataloader(dataset, config)
    config.knowledge_input_dim = dataset.knowledge_input_dim

    model = MemoryINP(config)
    loss_func = ELBOLoss()

    torch.manual_seed(config.seed)
    np.random.seed(config.seed)
    random.seed(config.seed)

    for i, batch in enumerate(train_dataloader):
        print(f"Batch {i}")
        context, target, knowledge, _ = batch

        x_context, y_context = context
        x_target, y_target = target

        if config.use_knowledge:
            outputs = model(x_context, y_context, x_target, y_target, knowledge)
        else:
            outputs = model(x_context, y_context, x_target, y_target, None)

        p_yCc, z_samples, q_z_Cc, q_zCct = outputs

        print(f"p_yCc mean shape: {p_yCc.mean.shape}")
        print(f"z_samples shape: {z_samples.shape}")
        print(f"q_z_Cc mean shape: {q_z_Cc.mean.shape}")
        print(f"q_zCct mean shape: {q_zCct.mean.shape}")

        loss = loss_func(outputs, y_target)

        print(f"Loss: {loss}\n")

        if i > 2:
            break
