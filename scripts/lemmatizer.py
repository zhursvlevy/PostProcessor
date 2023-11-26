import rootutils
rootutils.setup_root(__file__, indicator=".project-root", pythonpath=True)

from data.components.preprocessing import Processor
from tqdm import tqdm
import pandas as pd
from pathlib import Path

import click

@click.command()
@click.option("--path2data", help="path to pikabu parquet file")
@click.option("--pisave_dir", help="a directory to save processed parquet")
def process_data(path2data: str, save_dir: str) -> None:
    name = Path(path2data).stem
    data = pd.read_parquet(path2data)
    columns = ["title", "text_markdown"]
    processor = Processor()
    processed_data = {}
    for column in columns:
        processed = []
        rows = data[column].tolist()
        for row in tqdm(rows):
            processed.append(processor(row))
        processed_data[column] = processed
    processed_data["pluses"] = data["pluses"]
    processed_data["minuses"] = data["minuses"]
    processed_data["id"] = data["id"]
    processed_data["tags"] = data["tags"].apply(lambda x: ";".join(x.tolist()))
    pd.DataFrame(processed_data).to_parquet(Path(save_dir) / f"{name}_processed.parquet")

if __name__ == "__main__":
    process_data()