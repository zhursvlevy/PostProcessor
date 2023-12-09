import rootutils
rootutils.setup_root(__file__, indicator=".project-root", pythonpath=True)

from src.data.components.preprocessing import Processor
from tqdm import tqdm
import pandas as pd
from pathlib import Path
import os

import click

@click.command()
@click.option("--path2data", "-d", help="path to source parquet file")
@click.option("--save_dir", "-s", help="a directory to save processed parquet")
def process_post(path2data: str, save_dir: str) -> None:
    data = pd.read_parquet(path2data)
    columns = ["title", "text_markdown"]
    processor = Processor()
    for column in columns:
        processed = []
        rows = data[column].tolist()
        for row in tqdm(rows):
            processed.append(processor(row))
        data[column] = processed

    Path(save_dir).mkdir(parents=True, exist_ok=True)
    data.to_parquet(Path(save_dir) / f"lemmatized_texts.parquet")

if __name__ == "__main__":
    process_post()