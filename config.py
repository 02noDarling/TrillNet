import torch

SEQ_LEN = 48
CANDIDATE_SIZE = 1600
EMBED_DIM = 512
NHEAD = 64
WINDOW_SIZE = 20
ENCODER_NUMS = 2

#train
BATCH_SIZE = 50
EPOCHS = 200
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'