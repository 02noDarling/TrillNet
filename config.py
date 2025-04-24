import torch

SEQ_LEN = 48
CANDIDATE_SIZE = 1600
EMBED_DIM = 128
NHEAD = 16
WINDOW_SIZE = 6

#train
BATCH_SIZE = 50
EPOCHS = 200
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'