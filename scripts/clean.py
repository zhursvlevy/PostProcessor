import numpy as np
from pathlib import Path
import click
from ipynb_func import merge_dataset
from sklearn.model_selection import train_test_split
import json
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

URL_PATTERN = r'(https?:\/\/(?:www\.)?[-a-zA-Z0-9@:%._+~#=]{1,256}\.[a-zA-Z0-9()]{1,6}[-a-zA-Z0-9()@:%_+.~#?&/=]*)' 

@click.command()
@click.option("--data_path", "-d", default="data/pikabu", help="path to source data")
@click.option("--save_path", "-s", default="data/source", help="path to save data")
def clean_and_split(data_path: str, save_path: str) -> None:
    data_path = Path(data_path)
    logger.info("load data...")
    data = merge_dataset([data_path / "{}.parquet".format(i) for i in range(10)])
    logger.info("data successfully loaded!")
    logger.info("clean data...")
    images_or_videos = ["image" in row["blocks"]["type"] or "video" in row["blocks"]["type"] \
                        for i, row in data.iterrows()]
    text_data = data[~np.bool8(images_or_videos)]
    del data
    text_data["text_markdown"] = text_data["text_markdown"].str.replace(URL_PATTERN, '')
    text_data = text_data[~(((text_data['minuses'] == 0) & (text_data['pluses'] == 0)) \
                            | (text_data['minuses'] < 0) | (text_data['pluses'] < 0))]
    text_data.drop(columns=["blocks", "comments"], inplace=True)
    logger.info("data cleaned!")

    logger.info("save data...")
    save_path = Path(save_path)
    save_path.mkdir(exist_ok=True, parents=True)
    indexes = text_data["id"].tolist()
    train, test = train_test_split(indexes, test_size=0.1, shuffle=True, random_state=42)
    train, val = train_test_split(train, test_size=0.1, shuffle=True, random_state=42)
    text_data.to_parquet(save_path / "texts.parquet")
    with open(save_path / "indexes.json", "w") as f:
        json.dump({
            "train": train,
            "val": val,
            "test": test
        }, f)
    logger.info("data saved!")

if __name__ == "__main__":
    clean_and_split()