import torch
import torch.nn as nn
import torch.nn.functional as F
from flcore.trainmodel.moe.gate import Gating, CNNGating
from flcore.trainmodel.models import fastText
import numpy as np

# This class contains the UCB logic for expert selection
class UCBExperts:
    def __init__(self, num_experts, exploration_constant=1.0):
        self.num_experts = num_experts
        self.exploration_constant = exploration_constant
        self.counts = np.zeros(num_experts)  # N(a) in UCB formula
        self.rewards = np.zeros(num_experts)  # Q(a) in UCB formula

    def select_experts(self, k, total_plays):
        ucb_values = np.zeros(self.num_experts)
        for i in range(self.num_experts):
            if self.counts[i] == 0:
                # Arm not yet explored, give it an infinite value to ensure it's picked
                ucb_values[i] = float('inf')
            else:
                average_reward = self.rewards[i] / self.counts[i]
                exploration_term = self.exploration_constant * np.sqrt(np.log(total_plays) / self.counts[i])
                ucb_values[i] = average_reward + exploration_term
        
        # Select the indices of the top-k experts based on UCB values
        selected_indices = np.argsort(ucb_values)[-k:]
        return selected_indices

    def update_state(self, selected_indices, rewards):
        for i, idx in enumerate(selected_indices):
            self.counts[idx] += 1
            self.rewards[idx] += rewards[i]

# The original MoE class (for reference)
class MoE(nn.Module):
    def __init__(self, trained_experts):
        super(MoE, self).__init__()
        self.experts = nn.ModuleList(trained_experts)
        num_experts = len(trained_experts)
        # Assuming all experts have the same input dimension
        input_dim = trained_experts[0].in_features
        self.gating = Gating(input_dim, num_experts)

    def forward(self, x):
        # Get the weights from the gating network
        weights = self.gating(x)

        # Calculate the expert outputs
        outputs = torch.stack([expert(x) for expert in self.experts], dim=2)

        # Adjust the weights tensor shape to match the expert outputs
        weights = weights.unsqueeze(1).expand_as(outputs)

        # Multiply the expert outputs with the weights and
        # sum along the third dimension
        return torch.sum(outputs * weights, dim=2)

# The original ToPMoE class with EDM (for reference)
class ToPMoE(nn.Module):
    def __init__(self, trained_experts, gate_input_dim, args):
        super().__init__()
        self.experts = nn.ModuleList(trained_experts)
        self.num_experts = len(self.experts)
        self.args = args
        self.energy_T = getattr(args, "energy_T", getattr(args, "energe_T", 1.0))

        # gating network -> logits over experts
        self.gating = nn.Linear(gate_input_dim, self.num_experts, bias=True)

        # clamp topk to valid range (>=1, <= num_experts)
        req_topk = int(getattr(args, "topk", 1))
        self.topk = max(1, min(req_topk, self.num_experts))

    def calConfidence(self, normalized_item):
        E_k = -normalized_item
        # F_T^k(v^k) = -T * log(sum(exp(-E^k / T)))
        exp_term = torch.exp(-E_k / self.energy_T)  # exp(-E^k / T)
        sum_exp = torch.sum(exp_term)  # sum(exp(-E^k / T))
        F_T_k = -self.energy_T * torch.log(sum_exp)  # -T * log(sum_exp)
        H_k = -F_T_k
        return H_k

    def elementwise_cosine_similarity(self, x, y):
        dot_product = x * y  
        x_norm = torch.abs(x)  
        y_norm = torch.abs(y)  
        cosine_sim = dot_product / (x_norm * y_norm + 1e-8)  
        return cosine_sim
    
    def forward(self, rep):
        """
        rep: [B, D]
        Returns: [B, C] (same shape each expert head produces)
        """
        if rep.dim() != 2:
            rep = rep.view(rep.size(0), -1)

        # [B, E]
        gate_logits = self.gating(rep)

        # temperature + softmax scores
        scores = F.softmax(gate_logits / float(self.energy_T), dim=-1)  # [B, E]

        # compute "global" top-k experts (shared across the batch) to avoid
        # per-sample indexing mismatches and shape gymnastics
        global_scores = scores.mean(dim=0)            # [E]
        k = min(self.topk, self.num_experts)          # safety
        keep = torch.topk(global_scores, k).indices   # [k]
        keep_list = keep.tolist()

        # run selected experts
        # Each expert_i(rep): [B, C]
        expert_outs = [self.experts[i](rep) for i in keep_list]   # list of k tensors [B, C]
        stacked = torch.stack(expert_outs, dim=1)                 # [B, k, C]

        # gather corresponding per-sample weights and normalize
        # scores[:, keep] -> [B, k]
        kept_weights = scores.index_select(dim=1, index=keep)     # [B, k]
        kept_weights = kept_weights / (kept_weights.sum(dim=1, keepdim=True) + 1e-8)

        # Weighted sum over experts -> [B, C]
        out = (stacked * kept_weights.unsqueeze(-1)).sum(dim=1)
        return out

class NormalToPMoE(nn.Module):
    def __init__(self, trained_experts, gate_input_dim, args):
        super(NormalToPMoE, self).__init__()
        self.experts = nn.ModuleList(trained_experts)
        num_experts = len(trained_experts)
        # Assuming all experts have the same input dimension
        self.gating = Gating(gate_input_dim, num_experts)
        self.k = args.topk

    def forward(self, x):
        # Get the weights from the gating network
        weights = self.gating(x.flatten(1))  # [10,20]
        
        weights_values, indices = torch.topk(weights, self.k, dim=-1, largest=True, sorted=True, out=None)
        
        # Calculate the expert
        results = []
        for i in range(x.size(0)): 
            expert_results = [self.experts[idx](x[i]) for idx in indices[i]]
            stacked_expert_results = torch.stack(expert_results) # [10,10] 
            results.append(stacked_expert_results)
            
        final_results = torch.stack(results)  
        weights_x = weights_values.unsqueeze(-1).expand_as(final_results)

        return torch.sum(final_results * weights_x, dim=1) 

# The ExtractorToPMoE class with the new UCB logic
class ExtractorToPMoE(nn.Module): 
    def __init__(self, trained_experts, gate_input_dim, args):
        super(ExtractorToPMoE, self).__init__()
        self.experts = nn.ModuleList(trained_experts)
        self.num_experts = len(trained_experts)
        self.k = args.topk
        self.args = args

        ucb_c = getattr(args, "ucb_c", 1.0)
        self.ucb_selector = UCBExperts(num_experts=self.num_experts, exploration_constant=ucb_c)

        
        # Gating network is still used for weighting
        self.gating = Gating(gate_input_dim, self.num_experts)

    # The forward method now accepts the indices of experts selected by the client
    def forward(self, x, selected_experts_indices):
        # Get the gating weights
        weights = self.gating(x.flatten(1))  # [batch_size, num_experts]
        
        # Filter weights based on selected experts from UCB
        weights_values = []
        for i in range(x.size(0)):
            weights_values.append(weights[i, selected_experts_indices])
        weights_values = torch.stack(weights_values)
        
        # Calculate the expert outputs for the selected experts
        results = []
        if isinstance(self.experts[0], fastText):
            for i in range(x.size(0)): 
                expert_results = []
                for idx in selected_experts_indices:
                    expert_output = self.experts[idx].fc1(x[i].unsqueeze(0))
                    h = self.experts[idx].fc(expert_output)
                    out = F.log_softmax(h, dim=1).flatten(0)
                    expert_results.append(out)
                stacked_expert_results = torch.stack(expert_results) 
                results.append(stacked_expert_results)
        else:
            for i in range(x.size(0)): 
                expert_results = [self.experts[idx](x[i].unsqueeze(0)).flatten(0) for idx in selected_experts_indices]
                stacked_expert_results = torch.stack(expert_results)
                results.append(stacked_expert_results)
            
        final_results = torch.stack(results)
        weights_x = weights_values.unsqueeze(-1).expand_as(final_results)

        return torch.sum(final_results * weights_x, dim=1)

    # These methods are called by the client to get and update the UCB state
    def get_ucb_selection(self, num_total_plays):
        return self.ucb_selector.select_experts(self.k, num_total_plays)

    def update_ucb_state(self, selected_experts_indices, rewards):
        self.ucb_selector.update_state(selected_experts_indices, rewards)

# Other classes from the original file (unmodified)
class ParamToPMoE(nn.Module):
    def __init__(self, trained_experts, args):
        super(ParamToPMoE, self).__init__()
        self.experts = trained_experts # nn.params
        num_experts = len(trained_experts)
        # Assuming all experts have the same input dimension
        exp_dim = trained_experts[0].shape[0]  # same shape with input data -- rep
        self.gating = Gating(exp_dim, num_experts)
        self.k = args.topk
        
    def forward(self, x):
        # Get the weights from the gating network
        weights = self.gating(x)  # [10,20]
        
        weights_values, indices = torch.topk(weights, self.k, dim=-1, largest=True, sorted=True, out=None) 
        
        # Calculate the expert
        results = []
        for i in range(x.size(0)): 
            expert_results = [self.experts[idx] for idx in indices[i]]
            stacked_expert_results = torch.stack(expert_results) # [10,10] 
            results.append(stacked_expert_results)
            
        final_results = torch.stack(results)  
        weights_x = weights_values.unsqueeze(-1).expand_as(final_results)

        return torch.sum(final_results * weights_x, dim=1) 


class PatchMoE(nn.Module):
    def __init__(self, trained_experts, data_type="cifar10"):
        super(PatchMoE, self).__init__()
        self.data_type = data_type
        self.experts = nn.ModuleList(trained_experts)
        self.num_experts = len(trained_experts)
        # Assuming all experts have the same input dimension
        self.input_dim = trained_experts[0].in_features * self.num_experts
        self.gating = Gating(self.input_dim, self.num_experts)
        self.trained_experts = trained_experts

    def forward(self, x):
        batchsize = x.shape[0]
        x_flattened = x.reshape(batchsize, -1) # 10, 3072
        
        # Get the weights from the gating network
        weights = self.gating(x_flattened)
        
        segments = [x_flattened[:, i * self.trained_experts[0].in_features:(i + 1) * self.trained_experts[0].in_features] for i in range(self.num_experts)]

        outputs = []
        for expert, segment in zip(self.experts, segments):
            
            output = expert(segment)
            
            outputs.append(output)
        # Calculate the expert outputs
        # outputs = torch.stack([expert(x) for expert in self.experts], dim=2)

        stacked_outputs = torch.stack(outputs, dim=2) 
        
        # Adjust the weights tensor shape to match the expert outputs
        weights = weights.unsqueeze(1)
        
        # Multiply the expert outputs with the weights and
        # sum along the third dimension
        weight_out = stacked_outputs * weights  
        weight_out_t = weight_out.transpose(-2, -1) 
        
        if self.data_type =="cifar10":
            return weight_out_t.flatten(1).view(batchsize, 3, 32, 32) 
        
        return torch.sum(stacked_outputs * weights, dim=2)


class PatchCNNMoE(nn.Module):
    def __init__(self, trained_experts, data_type="cifar10"):
        super(PatchCNNMoE, self).__init__()
        self.data_type = data_type
        self.experts = nn.ModuleList(trained_experts)
        self.num_experts = len(trained_experts)
        # Assuming all experts have the same input dimension
        self.input_dim = trained_experts[0].in_channels
        self.gating = CNNGating(self.input_dim, self.num_experts)
        self.trained_experts = trained_experts

    def split_tensor(self, x, n):

        assert len(x.shape) == 4, "Input tensor must be 4-dimensional"
        
        p, q = x.shape[-2], x.shape[-1]
        

        if p % n == 0:
            split_dim = -2  
            block_size = q // n
        elif q % n == 0:
            split_dim = -1  
            block_size = p // n
        else:
            raise ValueError(f"Neither dimension {p} nor {q} can be evenly divided by {n}")
        
        split_x = torch.chunk(x, chunks=n, dim=split_dim)
        
        return split_x
    
    def forward(self, x):
        batchsize = x.shape[0]
        split_x = self.split_tensor(x, self.num_experts)
        
        weights = self.gating(x)
        # top k
        
        outputs = []
        for expert, segment in zip(self.experts, split_x):
            
            output = expert(segment)
            
            outputs.append(output)
        # Calculate the expert outputs
        # outputs = torch.stack([expert(x) for expert in self.experts], dim=2)

        stacked_outputs = torch.stack(outputs, dim=2) 
        
        # Adjust the weights tensor shape to match the expert outputs
        weights = weights.unsqueeze(1)
        
        # Multiply the expert outputs with the weights and
        # sum along the third dimension
        weight_out = stacked_outputs * weights  
        weight_out_t = weight_out.transpose(-2, -1) 
        weight_out_t = weight_out_t.flatten(1)
        
        if self.data_type =="cifar10":
            return weight_out_t 
        
        return torch.sum(stacked_outputs * weights, dim=2)