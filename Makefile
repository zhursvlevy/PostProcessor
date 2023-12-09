data_source:
	wget -i scripts/datalinks.txt -P data/pikabu && python scripts/clean.py

lemmatize:
	python scripts/lemmatizer.py -d data/source/texts.parquet -s data/lemmatized