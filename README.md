# Broker FIX Embeddings
Goal of this repo is to find a way to generate broker connectivity software by reading the FIX spec.

Current challenge is that FIX specs tend to be larger than the model's token limit.

Experimenting with creating embeddings DBs locally to reduce the size of the data sent in teh prompt.

## Generate Embeddings
```
python main.py -g
```
## Ask questinos about the document
```
python main.py --prompt