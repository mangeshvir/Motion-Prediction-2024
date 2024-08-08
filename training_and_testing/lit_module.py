import lightning as pl
import torch
import torch.nn.functional as F

class LitModule(pl.LightningModule):
    def __init__(self, model: torch.nn.Module, number_of_features: int, sequence_length: int, past_sequence_length: int, future_sequence_length: int, batch_size: int, use_lstm: bool = False):
        """
        Initialize the Lightning module.
        """
        super().__init__()
        self.model = model
        self.number_of_features = number_of_features
        self.sequence_length = sequence_length
        self.past_sequence_length = past_sequence_length
        self.future_sequence_length = future_sequence_length
        self.batch_size = batch_size
        self.use_lstm = use_lstm

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.
        """
        return self.model(x.float())

    def training_step(self, batch: torch.Tensor, batch_idx: int) -> torch.Tensor:
        """
        Training step.
        """
        return self.step(batch, batch_idx, "training")

    def validation_step(self, batch: torch.Tensor, batch_idx: int) -> torch.Tensor:
        """
        Validation step.
        """
        return self.step(batch, batch_idx, "validation")

    def test_step(self, batch: torch.Tensor, batch_idx: int) -> torch.Tensor:
        """
        Test step.
        """
        return self.step(batch, batch_idx, "test")

    def step(self, batch: torch.Tensor, batch_idx: int, stage: str) -> torch.Tensor:
        """
        Main step function for training, validation, and test.
        """
        x, y = self.prep_data_for_step(batch)
        y_hat = self(x)
        
        # No need to reshape y_hat here, as it should already be in the correct shape
        loss = F.mse_loss(y_hat, y)
        
        self.log(f"{stage}_loss", loss)
        return loss

    def prep_data_for_step(self, batch: torch.Tensor) -> tuple:
        """
        Prepare data for step function.
        """
        if self.use_lstm:
            x, y = batch
        else:
            x = batch[:, :self.past_sequence_length, :].float()
            y = batch[:, self.past_sequence_length:, :].float()
        return x, y

    def configure_optimizers(self):
        """
        Configure optimizers and schedulers.
        """
        optimizer = torch.optim.Adam(
            self.parameters(),
            lr=1e-3,
            weight_decay=1e-3,
            eps=1e-5,
            fused=False,
            amsgrad=True
        )
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            factor=0.2,
            patience=3,
            threshold=1e-4,
            cooldown=2,
            eps=1e-6,
            verbose=True,
        )
        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "monitor": "training_loss",
                "frequency": 1,
                "strict": True
            }
        }
