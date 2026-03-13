# Ved 50M — training config for Kaggle T4 (16 GB VRAM)
#
# Architecture: 8 layers, 512 embed dim, 8 Q heads, 2 KV heads (GQA)
# Approximate params: ~50M
#
# Effective batch:  16 seqs × 1024 tokens × 16 grad-accum = 262 144 tokens/step
# Tokens budget:    15 000 steps × 262 144 = ~3.9B tokens  (~4-5 hrs on T4)
# Resume with init_from='resume' to keep training in subsequent sessions.

# I/O
out_dir    = 'out-ved-50m'
eval_interval        = 500
log_interval         = 10
eval_iters           = 100
always_save_checkpoint = True
init_from            = 'scratch'   # change to 'resume' for subsequent sessions

# wandb (optional — set to True and fill in project name to track runs)
wandb_log        = False
wandb_project    = 'ved'
wandb_run_name   = 'ved-50m-fineweb'

# data
dataset                    = 'fineweb_edu'
gradient_accumulation_steps = 16
batch_size                 = 16     # micro-batch
block_size                 = 1024

# model — ~50M params
n_layer   = 8
n_head    = 8
n_kv_head = 2      # GQA: 4× smaller KV cache vs full MHA
n_embd    = 512
dropout   = 0.0
bias      = False

# optimizer
learning_rate = 3e-4
max_iters     = 7600           # 2B tokens ÷ 262K tokens/step; ~4 hrs on T4/P100
weight_decay  = 0.1
beta1         = 0.9
beta2         = 0.95
grad_clip     = 1.0

# lr schedule — cosine decay with warmup
decay_lr      = True
warmup_iters  = 500
lr_decay_iters= 7600
min_lr        = 3e-5

# system
device  = 'cuda'
dtype   = 'bfloat16'   # T4 supports bfloat16 via autocast
compile = True
