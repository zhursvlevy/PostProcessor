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
@torch.no_grad()
def main(dataset_dir: str, model_path: str, output_dir: str) -> None:

    (Path(output_dir) / model_path).mkdir(parents=True, exist_ok=True)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = RegressionTransformer(model_path, 312, 512, 0.5).to(device)
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    dataset = pd.read_parquet(dataset_dir)
    model.eval()
    embeddings = []
    scores = []
    texts = []
    cnt = 0

    for i, row in tqdm(dataset.iterrows()):
        text = row["text_markdown"]
        inp = tokenizer.encode_plus(text,
                                    None,
                                    add_special_tokens=True,
                                    # max_length=512,
                                    padding=True, # or "max_length"
                                    return_token_type_ids=True,
                                    truncation=True,
                                    return_tensors="pt")
        last_hidden_state = model.encoder(input_ids=inp["input_ids"].to(device),
                                          attention_mask=inp["attention_mask"].to(device)
                                    ).last_hidden_state[:, 0, :].squeeze(0).cpu().numpy()
        print(last_hidden_state.shape, last_hidden_state.dtype)
        embeddings.append(last_hidden_state)
        scores.append(row["wilson_score"])
        texts.append(text)

    new_data = pd.DataFrame({"text_markdown": texts, "embedding": embeddings, "wilson_score": scores})
    new_data.to_parquet(Path(output_dir) / model_path / Path(dataset_dir).name)

if __name__ == "__main__":
    main()

