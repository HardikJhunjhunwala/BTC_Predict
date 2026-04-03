from dataclasses import dataclass


@dataclass
class Config:
    csv_path: str = "data/raw/btcusd_1-min_data.csv"
    out_dir: str = "outputs"
    horizon_minutes: int = 60
    sample_every_n_rows: int = 5
    test_size_ratio: float = 0.20
    val_size_ratio: float = 0.10
    random_state: int = 42
    csv_chunk_rows: int = 200_000
    show_progress: bool = True
    use_torch: bool = True
    torch_batch_size: int = 8192
    torch_epochs: int = 1
    torch_lr: float = 1e-3
    run_epoch_sweep: bool = True
    epoch_sweep_values: tuple[int, ...] = (4, 8, 12, 16, 32, 48, 64, 80, 96, 112, 128, 144, 160, 176, 192, 208, 224, 240, 256)
