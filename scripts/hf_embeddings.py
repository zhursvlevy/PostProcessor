from transformers import AutoTokenizer, AutoModel
import click
from pathlib import Path
import pandas as pd
import torch
import rootutils
from tqdm import tqdm

rootutils.setup_root(__file__, indicator=".project-root", pythonpath=True)
from src.models.components.transformer import RegressionTransformer


@click.command()
@click.option("--dataset_dir", "-d", help="dataset directory")
@click.option("--model_path", "-m", default="cointegrated/rubert-tiny2", help="hugging face text encoder")
@click.option("--output_dir", "-o", default="../data/embeddings", help="path to save embeddings")
@click.option("--prepend_title", "-t", default=False, help="prepend title to markdown")
@click.option("--max_seq_len", "-l", default=512, help="max length of sentense")
@torch.no_grad()
def main(dataset_dir: str, model_path: str, output_dir: str, prepend_title: bool = False, max_seq_len: int = 512) -> None:

    (Path(output_dir) / model_path).mkdir(parents=True, exist_ok=True)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = RegressionTransformer(model_path, 312, 512, 0.5).to(device)
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    dataset = pd.read_parquet(dataset_dir)
    model.eval()
    embeddings = []
    ids = []

    for i, row in tqdm(dataset.iterrows()):
        markdown = str(row["text_markdown"])
        markdown = markdown.split()
        if prepend_title:
            title = str(row["title"])
            title = title.split()
            text = " ".join(title + markdown)
        else:
            text = " ".join(markdown)
        inp = tokenizer.encode_plus(
            text,
            None,
            add_special_tokens=True,
            max_length=max_seq_len,
            pad_to_max_length=True,
            return_token_type_ids=True,
            truncation=True,
            return_tensors="pt"
        )
        last_hidden_state = torch.mean(model.encoder(input_ids=inp["input_ids"].to(device),
                                                     attention_mask=inp["attention_mask"].to(device)
                                    ).last_hidden_state, dim=1)
        embeddings.append(last_hidden_state.squeeze(0).cpu().numpy())
        ids.append(row["id"])

    new_data = pd.DataFrame({"id": ids, "embedding": embeddings})
    new_data.to_parquet(Path(output_dir) / model_path / Path(dataset_dir).name)

if __name__ == "__main__":
    main()

