# Sentiment Analysis using Hierarchical Attention with PyTorch
This project is an implementation of a bidirectional LSTM with word-level hierarchical attention (Yang et al., 2016) for classification using PyTorch.
The current code supports either the IMDB Movie Reviews Dataset (Maas et al., 2011), or the Stanford Sentiment Treebank (SST) (Socher et al., 2013) dataset for the model.

## Usage
### Setup
- Python 3.5.2
- PyTorch 1.0.1
- TorchText 0.3.1
- SpaCy 2.1.8

### Training
To train the model, run the following command (some default parameters shown). Model will be saved in a subdirectory within './runs', with parameter values included in the name.

`python run.py --train --dataset imdb --lr 0.001 --n_epochs 5`

### Testing
For testing a trained model, use the command below. If a model path is not specified, one using the default parameters will be automatically input.

`python run.py --no-train --logdir <path-to-model>`

## References:
Maas, Andrew L., et al. "Learning word vectors for sentiment analysis." Proceedings of the 49th annual meeting of the association for computational linguistics: Human language technologies-volume 1. Association for Computational Linguistics, 2011.

Socher, Richard, et al. "Recursive deep models for semantic compositionality over a sentiment treebank." Proceedings of the 2013 conference on empirical methods in natural language processing. 2013.

Yang, Zichao, et al. "Hierarchical attention networks for document classification." Proceedings of the 2016 conference of the North American chapter of the association for computational linguistics: human language technologies. 2016.
