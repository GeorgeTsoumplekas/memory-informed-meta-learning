import torch
import torch.nn as nn


class Memory(nn.Module):
    def __init__(self, config):
        super().__init__()

        self.memory_knowledge = nn.Parameter(
            torch.empty(config.knowledge_dim, config.memory_slots),
            requires_grad=False,
        )
        self.memory_understanding = nn.Parameter(
            torch.empty(config.understanding_dim, config.memory_slots),
            requires_grad=False,
        )

        nn.init.orthogonal_(self.memory_knowledge)
        nn.init.orthogonal_(self.memory_understanding)

        self.w_w = torch.zeros((config.knowledge_dim,), requires_grad=False)
        self.w_u = torch.zeros((config.understanding_dim,), requires_grad=False)
        self.w_lu = torch.zeros((config.knowledge_dim,), requires_grad=False)

        self.beta_param = nn.Parameter(torch.tensor(0.0), requires_grad=True)

        self.config = config

    def forward(self, k, u):
        # STEP 1: Calculate w_r based on cosine similarity between k and memory knowledge
        knowledge_similarity = torch.matmul(k, self.memory_knowledge) / (
            torch.norm(k, dim=-1, keepdim=True)
            * torch.norm(self.memory_knowledge, dim=0, keepdim=True)
        )  # [bs, 1, k_dim] @ [k_dim, m_slots] -> [bs, 1, m_slots]
        w_r = torch.softmax(knowledge_similarity, dim=-1)  # [bs, 1, m_slots]

        # STEP 2: Erase the least used understanding slot entry
        # TODO: Check again if I should update both memory_knowledge and memory_understanding
        min_idx = torch.argmin(self.w_u)
        self.memory_knowledge[:, min_idx] = 0
        self.memory_understanding[:, min_idx] = 0

        # STEP 3: Calculate the updated w_u
        self.w_u = self.config.memory_gamma * self.w_u + w_r + self.w_w

        # STEP 4: Calculate the updated w_w
        beta = torch.sigmoid(self.beta_param)
        self.w_w = beta * w_r + (1 - beta) * self.w_lu

        # STEP 5: Calculate the updated w_lu
        # TODO: Check again
        min_w_u = torch.min(self.w_u, dim=-1, keepdim=True)[0]  # [bs, 1, 1]
        self.w_lu = torch.where(
            self.w_u < min_w_u, torch.ones_like(self.w_u), torch.zeros_like(self.w_u)
        )

        # STEP 6: Update understanding entries in memory
        self.memory_understanding = self.memory_understanding + self.w_w * u

        # STEP 7: Obtain refined understanding value
        u_final = torch.matmul(w_r, self.memory_knowledge.t())

        # STEP 8: Update knowledge entries in memory
        self.memory_knowledge = self.memory_knowledge + self.w_u * k

        return u_final
