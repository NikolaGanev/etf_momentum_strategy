from dataclasses import dataclass
from typing import List


@dataclass(frozen=True)
class Settings:
    # Correlation windows: default + sensitivity checks
    corr_windows: List[int] = (60, 120, 252)

    # Which return column to use for correlations
    returns_col: str = "ret_1d"

    # Adjacency construction
    use_positive_correlations_only: bool = True

    # Output paths
    processed_data_path: str = "data/processed/market_data.parquet"
    graph_output_dir: str = "data/processed/graphs"