#Creating the Sparse MOE (After acquiring the expert selector weight matrix, the top k values are selectively 
# multiplied with the outputs from the corresponding top-k experts for a given token

class SparseMoE (nn.Module) :
    def __init__(self, n_embed, num_experts, top_k):
        super(SparseMoE, self).__init__()
        self. router = TopkRouter(n_embed, num_experts, top_k)
        self.experts = nn.ModuleList([ExpertNN(n_embed) for _ in range(num_experts)])
        self.top_k = top_k
    def forward(self, x):
        gating_output, indices = self. router(x)
        final_output = torch. zeros_like (x)
        # Reshape inputs for batch processing
        flat_x = x. view(-1, x.size(-1))
        flat_gating_output = gating_output.view(-1, gating_output.size(-1))
        # Process each expert in parallel
        for i, expert in enumerate(self.experts):
            expert_mask = (indices == i). any (dim=-1)
            flat_mask = expert_mask.view(-1)
            if flat_mask.any():
                expert_input = flat_x[flat_mask]
                expert_output = expert (expert_input)
                #Extract and apply gating scores
                gating_scores = flat_gating_output[flat_mask, i]. unsqueeze(1)
                weighted_output = expert_output * gating_scores

                final_output [expert_mask] += weighted_output. squeeze(1)

        return final_output
    

num_experts = 3
top_k = 2
n_embd = 8
dropout=0.1

mh_output = torch.randn(1, 4, n_embd) 
sparse_moe = SparseMoE (n_embd, num_experts, top_k)
final_output = sparse_moe(mh_output)
print ("Shape of the final output:", final_output. shape)
print (final_output)    