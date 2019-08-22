import torch
from torchtext import data, datasets
import numpy as np
from settings import Settings as s


class DataPreparer(object):
    """
    Class for preparing a train-valid-test split of the data.
    Can be either IMDB or SST dataset, depending on the input name.
    """
    def __init__(self, name):
        self.name = name
        self.max_vocab_size = 30000
        self.TEXT = data.Field(tokenize='spacy', batch_first=True,
                               init_token=s.SOS, eos_token=s.EOS,
                               pad_token=s.PAD, unk_token=s.UNK,
                               include_lengths=True)
        if s.args.dataset == 'imdb':
            self.LABEL = data.Field(dtype = torch.float, sequential=False)
        else:
            self.LABEL = data.Field(dtype = torch.long, sequential=False)

    def prepare_data(self):
        """
        Prepares the three data splits for input to the model.
        """
        print("\nLoading dataset...")
        if self.name == 'imdb':
            train, test = datasets.IMDB.splits(self.TEXT, self.LABEL)
            train, validation = train.split(random_state=np.random.seed(s.args.seed))
        elif self.name == 'sst':
            train, validation, test = datasets.SST.splits(self.TEXT, self.LABEL)
        else:
            raise Exception("Dataset name must be 'imdb' or 'sst'.")

        print("Training size: {}".format(len(train)))
        print("Validation size: {}".format(len(validation)))
        print("Test size: {}".format(len(test)))

        print("\nBuilding vocabulary...")
        self.TEXT.build_vocab(train, max_size=self.max_vocab_size,
                              vectors="glove.6B.300d",
                              unk_init=torch.Tensor.normal_)
        self.LABEL.build_vocab(train)

        print("\nPreparing dataset split...")
        train_iter, valid_iter, test_iter = data.BucketIterator.splits(
            (train, validation, test), batch_size=s.args.batch,
            device=s.device, sort_within_batch=True)

        return train_iter, valid_iter, test_iter
