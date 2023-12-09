from transformers import AutoTokenizer, AutoModel, pipeline
import click
from pathlib import Path
import pandas as pd
import torch
import rootutils
from tqdm import tqdm

rootutils.setup_root(__file__, indicator=".project-root", pythonpath=True)
from src.models.components.transformer import RegressionTransformer
import torch.nn.functional as F
from torch import Tensor
from transformers import XLMRobertaTokenizer, XLMRobertaModel


def average_pool(last_hidden_states: Tensor, attention_mask: Tensor) -> Tensor:
    last_hidden = last_hidden_states.masked_fill(~attention_mask[..., None].bool(), 0.0)
    return last_hidden.sum(dim=1) / attention_mask.sum(dim=1)[..., None]

@click.command()
@click.option("--dataset_dir", "-d", help="dataset directory")
@click.option("--model_path", "-m", default="d0rj/e5-small-en-ru", help="hugging face text encoder")
@click.option("--output_dir", "-o", default="../data/embeddings", help="path to save embeddings")
@torch.no_grad()
def main(dataset_dir: str, model_path: str, output_dir: str) -> None:

    # (Path(output_dir) / model_path).mkdir(parents=True, exist_ok=True)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    tokenizer = XLMRobertaTokenizer.from_pretrained(model_path, use_cache=False)
    model = XLMRobertaModel.from_pretrained(model_path, use_cache=False)

    dataset = pd.read_parquet(dataset_dir)
    model.eval()
    embeddings = []
    scores = []
    texts = []

    for i, row in tqdm(dataset.iterrows()):
        if i == 1: break
        text = "query: " + row["text_markdown"]
        batch_dict = tokenizer(text,
                               max_length=512, 
                               padding=True, 
                               truncation=True, 
                               return_tensors='pt')
        print(batch_dict)
        outputs = model(**batch_dict)
        outputs = average_pool(outputs.last_hidden_state, batch_dict['attention_mask'])
        outputs = F.normalize(outputs, p=2, dim=1)
        print(outputs.shape, outputs.dtype)
        # embeddings.append(outputs.detach().cpu().numpy())
        # scores.append(row["wilson_score"])
        # texts.append(text)

    # new_data = pd.DataFrame({"text_markdown": texts, "embedding": embeddings, "wilson_score": scores})
    # new_data.to_parquet(Path(output_dir) / model_path / Path(dataset_dir).name)

if __name__ == "__main__":
    main()

