import torch
import torch.nn as nn
import torch.nn.functional as F
from .swiglu import SwiGLU

class NoisyTopkRouter(nn.Module):
    def __init__(self,config):
            super().__init__()
            self.n_expert = config.n_experts
            self.top_k = config.top_k
            self.router = nn.Linear(config.emb_dim, config.n_experts, bias=False)
            self.noise_linear =nn.Linear(config.emb_dim, config.n_experts)
    
    def forward(self,x):
        logits = self.router(x)
        noise_logits = self.noise_linear(x)
        noise = torch.randn_like(logits)*F.softplus(noise_logits) 
        noisy_logits = logits + noise

        top_k_logits, top_k_indices =  noisy_logits.topk(self.top_k, dim=-1)
        zeros = torch.full_like(logits, float('-inf'))
        sparse_logits = zeros.scatter(-1, top_k_indices, top_k_logits)
        routing_output = torch.softmax(sparse_logits, dim=-1)
        return routing_output, top_k_indices
            

class MoE(nn.Module):
    def __init__(self,config):
        super().__init__()
        self.n_expert = config.n_experts
        self.top_k = config.top_k
        self.n_shared_experts = config.n_shared_experts
        self.experts = nn.ModuleList(SwiGLU(config.emb_dim, config.expert_dim, bias=False) for _ in range(self.n_expert))
        self.shared_exptets = nn.ModuleList(SwiGLU(config.emb_dim, config.expert_dim, bias=False) for _ in range(self.n_shared_experts))
        self.router = NoisyTopkRouter(config)


    def forward(self,x):
        ### shared expert
        shared_output = torch.zeros_like(x)
        for shared_expect in self.shared_exptets:
            shared_output +=  shared_expect(x)

        routing_output, indices = self.router(x)
        routed_output = torch.zeros_like(x)

        flat_x = x.view(-1,x.size(-1))
        flat_routing_output = routing_output.view(-1, routing_output.size(-1))

        for i,expert in enumerate(self.experts):
            expert_mask = (indices == i).any(dim=-1)
            flat_mask = expert_mask.view(-1)
            if flat_mask.any():
                expert_input = flat_x[flat_mask]
                expert_output = expert(expert_input)

                routed_scores = flat_routing_output[flat_mask, i].unsqueeze(-1)
                weighted_output = expert_output * routed_scores

                routed_output[expert_mask] += weighted_output
        routed_output = routed_output.view_as(x)
        final_output = shared_output +  routed_output
        return final_output

