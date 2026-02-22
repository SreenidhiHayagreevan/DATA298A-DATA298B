import s3fs
import pandas as pd
from config.dataset_paths import DATASET_PATHS, AWS_REGION


class DatasetLoader:

    def __init__(self):
        print("Initializing S3 filesystem...")
        self.fs = s3fs.S3FileSystem(
            client_kwargs={"region_name": AWS_REGION}
        )

    def load(self, name: str):

        if name not in DATASET_PATHS:
            raise ValueError(f"Dataset {name} not configured.")

        path = DATASET_PATHS[name]
        print(f"Loading {name} from {path}")

        df = pd.read_parquet(path, filesystem=self.fs)

        print(f"Loaded {name}: {len(df)} rows")
        return df

    def load_all(self):
        datasets = {}
        for name in DATASET_PATHS:
            datasets[name] = self.load(name)
        return datasets
