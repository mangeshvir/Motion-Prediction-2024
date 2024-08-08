import os
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset
from torchvision import transforms

class inD_RecordingDataset(Dataset):
    def __init__(self, path, recording_id, sequence_length, features, past_sequence_length, future_sequence_length, train=True, use_lstm=False):
        """
        Initialize dataset.
        """
        super().__init__()
        self.path = path
        self.recording_id = recording_id
        self.sequence_length = sequence_length
        self.features = features
        self.past_sequence_length = past_sequence_length
        self.future_sequence_length = future_sequence_length
        self.train = train
        self.use_lstm = use_lstm
        self.transform = self.get_transform()
        self.data = self.load_data()

    def load_data(self):
        """
        Load data from CSV files.
        """
        data_frames = []

        if isinstance(self.recording_id, list):
            for rid in self.recording_id:
                file_path = f"{self.path}/{rid}_tracks.csv"
                if os.path.exists(file_path):
                    data_frames.append(pd.read_csv(file_path, delimiter=',', header=0, usecols=self.features, dtype='float32'))
                else:
                    raise FileNotFoundError(f"File {file_path} not found.")
        else:
            file_path = f"{self.path}/{self.recording_id}_tracks.csv"
            if os.path.exists(file_path):
                data_frames.append(pd.read_csv(file_path, delimiter=',', header=0, usecols=self.features, dtype='float32'))
            else:
                raise FileNotFoundError(f"File {file_path} not found.")

        return pd.concat(data_frames, ignore_index=True)

    def __len__(self):
        """
        Return dataset length.
        """
        return len(self.data) - self.sequence_length

    def __getitem__(self, idx: int):
        """
        Get item by index.
        """
        if idx < 0 or idx >= self.__len__():
            raise IndexError(f"Index {idx} is out of bounds for dataset with length {self.__len__()}.")

        data = self.data.iloc[idx:idx + self.sequence_length]

        if self.transform:
            data = self.transform(np.array(data, dtype='float32')).squeeze()

        if self.use_lstm:
            inputs = torch.tensor(data[:self.past_sequence_length], dtype=torch.float32)
            targets = torch.tensor(data[self.past_sequence_length:], dtype=torch.float32)
            return inputs, targets
        else:
            return data

    def get_transform(self):
        """
        Get data transform.
        """
        return transforms.Compose([
            transforms.ToTensor(),
        ])
