"""MFL trainer package: dataset wiring, per-client training, FedAvg server, CLI."""
from .dataset import (
    LABEL_COLS,
    PartitionPaths,
    PairedCXRDataset,
    TokenizingCollator,
    build_client_loaders,
    build_global_eval_loaders,
    load_partition,
    load_prepared_csv,
    make_tokenizer,
)
from .client import ClientConfig, local_train
from .server import FederatedServer, ServerConfig, evaluate, fedavg_state_dicts

__all__ = [
    "LABEL_COLS",
    "PartitionPaths",
    "PairedCXRDataset",
    "TokenizingCollator",
    "build_client_loaders",
    "build_global_eval_loaders",
    "load_partition",
    "load_prepared_csv",
    "make_tokenizer",
    "ClientConfig",
    "local_train",
    "FederatedServer",
    "ServerConfig",
    "evaluate",
    "fedavg_state_dicts",
]
