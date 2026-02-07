# hyperparameters to be used in the training 
batch_size = 16 # how many independent sequences will we process in parallel?
block_size = 32 # what is the maximum context length for predictions?
max_iters = 20
eval_interval = 100
learning_rate = 1e-3
eval_iters = 400
head_size = 16
n_embed = 128
n_head = 8
n_layer = 8
dropout = 0.1
num_experts = 8
top_k =2 