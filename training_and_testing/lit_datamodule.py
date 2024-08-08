import lightning as pl
import torch
from torch.utils.data import DataLoader, random_split
import os
from lit_dataset import inD_RecordingDataset

class inD_RecordingModule(pl.LightningDataModule):
    """
    LightningDataModule for inD dataset.
    """
    def __init__(self, data_path, recording_id, sequence_length, past_sequence_length, future_sequence_length, features, batch_size: int = 32, use_lstm: bool = False):
        """
        Initialize the data module.
        """
        super().__init__()
        self.data_path = data_path
        self.recording_id = recording_id
        self.batch_size = batch_size
        self.sequence_length = sequence_length
        self.past_sequence_length = past_sequence_length
        self.future_sequence_length = future_sequence_length
        self.features = features
        self.use_lstm = use_lstm
        self.save_hyperparameters()

    def setup(self, stage: str):
        """
        Setup the data for different stages.
        """
        if stage == "test":
            self.test = inD_RecordingDataset(self.data_path, self.recording_id, self.sequence_length, self.features, self.past_sequence_length, self.future_sequence_length, use_lstm=self.use_lstm)
        elif stage == "predict":
            self.predict = inD_RecordingDataset(self.data_path, self.recording_id, self.sequence_length, self.features, self.past_sequence_length, self.future_sequence_length, use_lstm=self.use_lstm)
        elif stage == "fit":
            full_dataset = inD_RecordingDataset(self.data_path, self.recording_id, self.sequence_length, self.features, self.past_sequence_length, self.future_sequence_length, use_lstm=self.use_lstm)
            train_size = int(0.80 * len(full_dataset))
            val_size = len(full_dataset) - train_size
            self.train, self.val = random_split(full_dataset, [train_size, val_size])

    def train_dataloader(self):
        """
        Train dataloader.
        """
        return DataLoader(self.train, batch_size=self.batch_size, num_workers=0, persistent_workers=False, shuffle=True, pin_memory=True, collate_fn=self.collate_fn if self.use_lstm else None)

    def val_dataloader(self):
        """
        Validation dataloader.
        """
        return DataLoader(self.val, batch_size=self.batch_size, num_workers=0, persistent_workers=False, shuffle=False, pin_memory=True, collate_fn=self.collate_fn if self.use_lstm else None)

    def test_dataloader(self):
        """
        Test dataloader.
        """
        return DataLoader(self.test, batch_size=self.batch_size, num_workers=0, pin_memory=True, collate_fn=self.collate_fn if self.use_lstm else None)

    def predict_dataloader(self):
        """
        Predict dataloader.
        """
        return DataLoader(self.predict, batch_size=self.batch_size, num_workers=0, pin_memory=True, collate_fn=self.collate_fn if self.use_lstm else None)

    def teardown(self, stage: str):
        """
        Clean-up when the run is finished.
        """
        print("Teardown complete.")
        
    def collate_fn(self, batch):
        """
        Custom collate function for LSTM.
        """
        inputs = torch.stack([item[0] for item in batch])
        targets = torch.stack([item[1] for item in batch])
        return inputs, targets
