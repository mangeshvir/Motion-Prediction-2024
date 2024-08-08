from lightning.pytorch.callbacks import LearningRateMonitor, ModelCheckpoint

def create_callbacks():
    """
    Create a list of PyTorch Lightning callbacks.
    
    Returns
    -------
    list
        List of callbacks including LearningRateMonitor and ModelCheckpoint.
    """
    callbacks = [
        LearningRateMonitor(logging_interval='step'),
        
        # ModelCheckpoint for training loss
        ModelCheckpoint(
            monitor="training_loss",
            filename="TRAIN_CKPT_{training_loss:.2f}-{validation_loss:.2f}-{epoch}",
            mode="min",
            every_n_epochs=1,
            save_top_k=2,
            verbose=True,
            auto_insert_metric_name=True,
            save_on_train_epoch_end=True
        ),
        
        # ModelCheckpoint for validation loss
        ModelCheckpoint(
            monitor="validation_loss",
            filename="VAL_CKPT_{validation_loss:.2f}-{training_loss:.2f}-{epoch}",
            mode="min",
            every_n_epochs=1,
            save_top_k=2,
            verbose=True,
            auto_insert_metric_name=True
        ),
        
        # Periodic ModelCheckpoint every 10 epochs
        ModelCheckpoint(
            filename="REC_CKPT_{epoch}-{validation_loss:.2f}-{training_loss:.2f}",
            every_n_epochs=10,
            verbose=True,
            auto_insert_metric_name=True
        )
    ]
    return callbacks
