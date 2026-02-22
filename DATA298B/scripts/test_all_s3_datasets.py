from data.load_s3_datasets import S3DatasetLoader

if __name__ == "__main__":
    loader = S3DatasetLoader()

    datasets = loader.load_all()

    for name, df in datasets.items():
        print(f"\n===== {name.upper()} =====")
        print(df.head())
        print("Rows:", len(df))
