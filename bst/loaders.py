import torch
import numpy as np

import os
import ast
import io
import csv

import pandas as pd
from torch.utils.data import Dataset
import pickle


class ShuffledCSVDataset(Dataset):
    """
    Memory-efficient dataset that reads CSV rows on-demand with shuffling support
    Supports train/val/test split with separate caching
    """
    def __init__(self, csv_file, split='train', train_ratio=0.8, val_ratio=0.1, test_ratio=0.1, 
                 shuffle=True, cache_file=None, seed=42, movie_vocab_stoi=None, user_vocab_stoi=None, obj=None):
        self.csv_file = csv_file
        self.split = split
        self.train_ratio = train_ratio
        self.val_ratio = val_ratio
        self.test_ratio = test_ratio
        self.shuffle = shuffle
        self.cache_file = cache_file or f"{csv_file}.cache"
        self.seed = seed
        self.obj = obj

        self.movie_vocab_stoi = movie_vocab_stoi
        self.user_vocab_stoi = user_vocab_stoi
        
        # Validate split ratios
        assert abs(train_ratio + val_ratio + test_ratio - 1.0) < 1e-6, "Split ratios must sum to 1.0"
        assert split in ['train', 'val', 'test'], "Split must be 'train', 'val', or 'test'"
        
        # Get dataset info
        self._get_dataset_info()
        
        # Create or load split indices
        self._create_split_indices()
    
    def _get_dataset_info(self):
        """Get column info and precompute row offsets"""
        print("Analyzing CSV file...")
        if self.split=='train':
            with open(self.csv_file, 'rb') as f:
                header = f.readline().decode()
                self.columns = header.strip().split(',')
                
                self.offsets = []
                offset = f.tell()
                for line in f:
                    self.offsets.append(offset)
                    offset += len(line)

            self.n_features = len(self.columns)
            self.total_rows = len(self.offsets)

            print(f"Dataset info:")
            print(f"  Rows: {self.total_rows:,}")
            print(f"  Columns: {self.n_features}")
            print(f"  Features: {self.columns[:5]}..." if len(self.columns) > 5 else f"  Features: {self.columns}")


        else:
            self.offsets = self.obj.offsets
        # Preview object columns using a few rows
        preview = pd.read_csv(self.csv_file, nrows=1000)
        self.object_columns = preview.select_dtypes(include=['object']).columns.tolist()
        if self.object_columns:
            print(f"  Object columns detected: {self.object_columns}")

    def _create_split_indices(self):
        """Create and cache train/val/test split indices"""
        indices_file = f"{self.cache_file}_split_indices.pkl"
        
        if os.path.exists(indices_file):
            try:
                print("Loading cached split indices...")
                with open(indices_file, 'rb') as f:
                    splits = pickle.load(f)
                if self.split == 'train':
                    self.train_indices = splits['train']
                elif self.split == 'val':
                    self.val_indices = splits['val']
                else:  # test
                    self.test_indices = splits['test']
            except (EOFError, pickle.UnpicklingError) as e:
                print(f"Cache file {indices_file} is corrupted ({e}). Recreating...")
                os.remove(indices_file)
                return self._create_split_indices()  # Retry cleanly
        else:
            print("Creating train/val/test split indices...")
            
            # Set seed for reproducible splits
            np.random.seed(self.seed)
            
            # Create shuffled indices for the entire dataset
            all_indices = np.random.permutation(self.total_rows)
            
            # Calculate split sizes
            train_size = int(self.total_rows * self.train_ratio)
            val_size = int(self.total_rows * self.val_ratio)
            test_size = self.total_rows - train_size - val_size
            
            # Split indices
            self.train_indices = all_indices[:train_size]
            self.val_indices = all_indices[train_size:train_size + val_size]
            self.test_indices = all_indices[train_size + val_size:]
            
            # Additional shuffling within each split if requested
            if self.shuffle:
                np.random.shuffle(self.train_indices)
                np.random.shuffle(self.val_indices)
                np.random.shuffle(self.test_indices)
            
            # Cache the splits
            splits = {
                'train': self.train_indices,
                'val': self.val_indices,
                'test': self.test_indices
            }
            with open(indices_file, 'wb') as f:
                pickle.dump(splits, f)
            
            print(f"Split created:")
            print(f"  Train: {len(self.train_indices):,} samples ({len(self.train_indices)/self.total_rows*100:.1f}%)")
            print(f"  Val: {len(self.val_indices):,} samples ({len(self.val_indices)/self.total_rows*100:.1f}%)")
            print(f"  Test: {len(self.test_indices):,} samples ({len(self.test_indices)/self.total_rows*100:.1f}%)")
            print(f"Cached splits to {indices_file}")
        
        # Set current split indices
        if self.split == 'train':
            self.indices = self.train_indices
        elif self.split == 'val':
            self.indices = self.val_indices
        else:  # test
            self.indices = self.test_indices
        
        print(f"Using {self.split} split: {len(self.indices):,} samples")
        
    def _preprocess_row(self, row_data):
        """Preprocess a single row (handle object columns)"""
        for col in self.object_columns:
            if col in row_data.columns:
                # Simple preprocessing: convert to category codes
                # You can customize this based on your data
                try:
                    row_data[col] = row_data[col].apply(ast.literal_eval)
                except:
                    row_data[col] = 0  # Default value for problematic data
        
        return row_data
    
    def __len__(self):
        return len(self.indices)  # Return length of current split, not total rows
    
    def __getitem__(self, idx):
        def parse_csv_line(line: str):
            return next(csv.reader([line], skipinitialspace=True))
        
        actual_idx = self.indices[idx]
        offset = self.offsets[actual_idx]
        try:
            with open(self.csv_file, 'r') as f:
                f.seek(offset)
                line = f.readline().strip()
                line = parse_csv_line(line)

            user_id = line[0]
            try:
                movie_sequence = ast.literal_eval(line[1])
            except:
                movie_sequence = []

            movie_data = [self.movie_vocab_stoi.get(item, 0) for item in movie_sequence]
            user_data = self.user_vocab_stoi.get(str(user_id), 0)

            return torch.tensor(movie_data), torch.tensor(user_data)
        
        except Exception as e:
            return self.__getitem__((idx+1) % len(self))  # you may replace with a while loop to avoid recursion