# PostProcessor
A service for ranking and tagging posts on social networks

## Работа с репозиторием
Клонируем репо:
```console
git clone https://github.com/zhursvlevy/PostProcessor.git
```
Затем делаем новую ветку из мастера в таком формате:
```console
git branch feature/contributor_name
```
Поработали в ветке, сделали коммит и пушите в репозиторий:
```console
git push origin feature/contributor_name:feature/contributor_name
```

## Download data
```console
wget -i scripts/datalinks.txt -P data/pikabu
```

## Train pytorch modesls
This project uses lightning for code structuring and hydra for flexible experiment configuring and tracking.
All models can be implemented in uniform structure. Models should be implemented in src/models/components. 
Training settings must be implemented in src/models. All data processing functional should be implemented in
src/data/components and lightning data modules must be implemented at src/data. All model configurations is set
in configs folder. You can configure model in .yaml files or set parameters in terminal:
```console
python src/train.py "hydra parameters overrides"
```
## Train pytorch models. Example
```console
python src/train.py experiment=rate_prediction model.net.hidden_dim=256 model.optimizer.lr=1e-5
```