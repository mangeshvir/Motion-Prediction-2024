import os
import datetime
from lightning.pytorch.loggers import WandbLogger
import torch
from torch import nn
from pathlib import Path
import sys

def get_data_path() -> tuple:
    """
    Get data and log paths.
    """
    project_path = Path(__file__).resolve().parents[1]
    data_directory = Path("C:/Users/Prasa/OneDrive/Desktop/code/code/dataset/data")
    data_path = project_path / data_directory
    sys.path.append(str(data_path))

    current_file_path = Path(__file__).parent.resolve()
    log_directory = 'logs'
    log_path = current_file_path / log_directory
    
    return data_path, log_path

def create_wandb_logger(log_path: Path, project_name: str, stage: str) -> WandbLogger:
    """
    Create a WandB logger.
    """
    date_str = get_current_datetime_str()
    wandb_logger = WandbLogger(
        project=project_name,
        name=f"{project_name}_{stage}_{date_str}",
        log_model=False,
        version=date_str,
        save_dir=str(log_path / "wandb_logs")
    )
    return wandb_logger

def get_current_datetime_str() -> str:
    """
    Get current datetime as string.
    """
    now = datetime.datetime.now()
    return now.strftime("%Y_%m_%d__%H_%M_%S")

def build_module(hidden_size: int, output_size: int, num_layers: int, pre_layers: list = None, dropout_rate: float = 0.3) -> nn.ModuleList:
    """
    Build a neural network module.
    """
    layer_dims = torch.linspace(hidden_size, output_size, num_layers).int()
    layers = nn.ModuleList()
    
    if pre_layers:
        layers.extend(pre_layers if isinstance(pre_layers, list) else [pre_layers])
    
    for i in range(len(layer_dims) - 1):
        layers.append(nn.Linear(layer_dims[i].item(), layer_dims[i + 1].item()))
        layers.append(nn.Dropout(dropout_rate))
        layers.append(nn.Softplus())
    
    layers.append(nn.Linear(layer_dims[-1].item(), output_size))
    
    return layers