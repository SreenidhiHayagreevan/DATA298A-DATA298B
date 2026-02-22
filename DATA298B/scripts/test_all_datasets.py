from data.dataset_loader import DatasetLoader

loader = DatasetLoader()
datasets = loader.load_all()

for name, df in datasets.items():
    print("\n==========", name.upper(), "==========")
    print(df.head())
    print("Rows:", len(df))
