import math
import os
import tqdm

import torch
from torch import nn, Tensor
from torchtext.vocab import vocab
from torch.utils.data import DataLoader

from collections import Counter

import pandas as pd
import time

from model import TransformerModel
from loaders import ShuffledCSVDataset


csv_file = '/home/sd8/Coding/exp/recommendations/data-created-bst/csv/total.csv'
movie_file = '/home/sd8/Coding/exp/recommendations/data-created/movies.csv'

df = pd.read_csv(csv_file, usecols=['user_id'])
movies = pd.read_csv(movie_file)

# Genarting a list of unique movie ids
movie_ids = movies.movie_id.unique()
movie_ids = list(map(lambda x: str(x), movie_ids))

# Counter is used to feed movies to movive_vocab
movie_counter = Counter(movie_ids)
# Genarting vocabulary
movie_vocab = vocab(movie_counter, specials=['<unk>'])
# For indexing input ids
movie_vocab_stoi = movie_vocab.get_stoi()

# Movie to title mapping dictionary
movie_title_dict = dict(zip(movies.movie_id, movies.title))


# Similarly generating a vocabulary for user ids
user_ids = df.user_id.unique()
user_ids = list(map(lambda x: str(x), user_ids))
user_counter = Counter(user_ids)
user_vocab = vocab(user_counter, specials=['<unk>'])
user_vocab_stoi = user_vocab.get_stoi()

del df, movies

# region Dataloader

# These three will share the same cached split indices
train_dataset = ShuffledCSVDataset(csv_file, split='train', seed=42, 
                                   user_vocab_stoi=user_vocab_stoi, 
                                   movie_vocab_stoi=movie_vocab_stoi)

val_dataset = ShuffledCSVDataset(csv_file, split='val', seed=42, 
                                 user_vocab_stoi=user_vocab_stoi, 
                                 movie_vocab_stoi=movie_vocab_stoi, obj=train_dataset)

batch_size = 2048
shuffle_dataloader = True
num_workers = 7

train_dataloader = DataLoader(
    train_dataset,
    batch_size=batch_size,
    shuffle=shuffle_dataloader,  # Only shuffle training data
    num_workers=num_workers,
    pin_memory=False,
    persistent_workers=True if num_workers > 0 else False
)

val_dataloader = DataLoader(
    val_dataset,
    batch_size=batch_size,
    shuffle=shuffle_dataloader,  # Only shuffle training data
    num_workers=num_workers,
    pin_memory=False,
    persistent_workers=True if num_workers > 0 else False
)
# end

# region MODEL INIT

ntokens = len(movie_vocab)  # size of vocabulary
nusers = len(user_vocab)
emsize = 128  # embedding dimension
d_hid = 128  # dimension of the feedforward network model
nlayers = 2  # number of ``nn.TransformerEncoderLayer``
nhead = 2  # number of heads in ``nn.MultiheadAttention``
dropout = 0.2  # dropout probability

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = TransformerModel(ntokens, nusers, emsize, nhead, d_hid, nlayers, dropout).to(device)

criterion = nn.CrossEntropyLoss()
lr = 1.0  # learning rate
optimizer = torch.optim.SGD(model.parameters(), lr=lr)
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, 1.0, gamma=0.95)
# end

# region TRAIN

def train(model: nn.Module, train_iter, epoch) -> None:
    # Switch to training mode
    model.train()
    total_loss = 0.
    log_interval = 1000
    start_time = time.time()

    for i, (movie_data, user_data) in tqdm.tqdm(enumerate(train_iter)):
        # Load movie sequence and user id
        movie_data, user_data = movie_data.to(device), user_data.to(device)
        user_data = user_data.reshape(-1, 1)

        # Split movie sequence to inputs and targets
        inputs, targets = movie_data[:, :-1], movie_data[:, 1:]
        targets_flat = targets.reshape(-1)

        # Predict movies
        output = model(inputs, user_data)
        output_flat = output.reshape(-1, ntokens)

        # Backpropogation process
        loss = criterion(output_flat, targets_flat)
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 0.5)
        optimizer.step()

        total_loss += loss.item()
        # Results
        if i % log_interval == 0 and i > 0:
            lr = scheduler.get_last_lr()[0]
            ms_per_batch = (time.time() - start_time) * 1000 / log_interval
            cur_loss = total_loss / log_interval
            ppl = math.exp(cur_loss)
            print(f'| epoch {epoch:3d} '
                  f'lr {lr:02.2f} | ms/batch {ms_per_batch:5.2f} | '
                  f'loss {cur_loss:5.2f} | ppl {ppl:8.2f}')
            total_loss = 0
            start_time = time.time()
            torch.cuda.empty_cache()


def evaluate(model: nn.Module, eval_data: Tensor) -> float:
    # Switch the model to evaluation mode.
    # This is necessary for layers like dropout,
    model.eval()
    total_loss = 0.0

    with torch.no_grad():
        for i, (movie_data, user_data) in enumerate(eval_data):
            # Load movie sequence and user id
            movie_data, user_data = movie_data.to(device), user_data.to(device)
            user_data = user_data.reshape(-1, 1)
            # Split movie sequence to inputs and targets
            inputs, targets = movie_data[:, :-1], movie_data[:, 1:]
            targets_flat = targets.reshape(-1)
            # Predict movies
            output = model(inputs, user_data)
            output_flat = output.reshape(-1, ntokens)
            # Calculate loss
            loss = criterion(output_flat, targets_flat)
            total_loss += loss.item()
    return total_loss / (len(eval_data) - 1)

best_val_loss = float('inf')
epochs = 10
temp_dir = './model_files'
best_model_params_path = os.path.join(temp_dir, "best_model_params.pt")

for epoch in range(1, epochs + 1):
    epoch_start_time = time.time()

    # Training
    train(model, train_dataloader, epoch)

    # Evaluation
    val_loss = evaluate(model, val_dataloader)

    # Compute the perplexity of the validation loss
    val_ppl = math.exp(val_loss)
    elapsed = time.time() - epoch_start_time

    # Results
    print('-' * 89)
    print(f'| end of epoch {epoch:3d} | time: {elapsed:5.2f}s | '
        f'valid loss {val_loss:5.2f} | valid ppl {val_ppl:8.2f}')
    print('-' * 89)

    # Save best model
    if val_loss < best_val_loss:
        best_val_loss = val_loss
        torch.save(model.state_dict(), best_model_params_path)

    scheduler.step()

model.load_state_dict(torch.load(best_model_params_path)) # load best model states
