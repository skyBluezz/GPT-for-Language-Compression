# train a miniature character-level shakespeare model
# good for debugging and playing on macbooks and such

out_dir = 'out-enwik8'
eval_interval = 250 # keep frequent because we'll overfit
eval_iters = 200
log_interval = 10 # don't print too too often

# we expect to overfit on this small dataset, so only save when val improves
always_save_checkpoint = False

wandb_log = False # override via command line if you like
wandb_project = 'enwik8'
wandb_run_name = 'mini-gpt'

dataset = 'enwik8'
gradient_accumulation_steps = 6
batch_size = 32
block_size = 256*4 # context of up to 256 previous characters

# baby GPT model :)
n_layer = 8
n_head = 8
n_embd = 64*n_head
dropout = 0.2

learning_rate = 1e-3 #6e-4 # with baby networks can afford to go a bit higher
max_iters = 10000
lr_decay_iters = 15000 # make equal to max_iters usually
min_lr = 1e-4 # learning_rate / 10 usually
beta2 = 0.99 # make a bit bigger because number of tokens per iter is small

warmup_iters = 100 # not super necessary potentially

vowel_cons_embed_weight = 0.01
future_window = 1#2 # Try to challenge the model to look further into the future
perplexity_loss_weight = 0#2e-3

super_window: int = 1024
kernel_size: int = 256