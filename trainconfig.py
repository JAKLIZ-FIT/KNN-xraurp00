from dataclasses import dataclass

@dataclass
class TrainConfig:
    num_checkpoints : int
    epochs: int 
    use_gpu: bool
    save_path: Path
    early_stop_threshold: float

    