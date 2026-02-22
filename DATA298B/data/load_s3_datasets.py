import pandas as pd
import s3fs
from config.s3_config import AWS_REGION, S3_DATASETS


class S3DatasetLoader:
    def __init__(self):
        print("Initializing S3 filesystem...")
        self.fs = s3fs.S3FileSystem(
            client_kwargs={"region_name": AWS_REGION}
        )

    def load_dataset(self, dataset_name: str) -> pd.DataFrame:
        if dataset_name not in S3_DATASETS:
            raise ValueError(f"Dataset '{dataset_name}' not found in config.")

        path = S3_DATASETS[dataset_name]
        print(f"Loading dataset from {path}")

        df = pd.read_parquet(
            path,
            filesystem=self.fs
        )

        print(f"Loaded {dataset_name}: {len(df)} rows")
        return df

    def load_all(self) -> dict:
        datasets = {}
        for name in S3_DATASETS:
            datasets[name] = self.load_dataset(name)
        return datasets
