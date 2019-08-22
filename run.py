import os, pickle
import torch
import torch.optim as optim
import torch.nn as nn
import numpy as np
import argparse
from model import Network
from data import DataPreparer
from settings import Settings as s

def train_mod(model, train_iter, optimizer, criterion, epoch_num):
    """
    Trains the model for one epoch with the given random seed.
    """
    train_loss = 0
    train_results = [0, 0, 0, 0]

    Preds, Labels = [], []

    model.train()

    for idx, batch in enumerate(train_iter):
        optimizer.zero_grad()

        x, seqlens = batch.text

        y = batch.label - 1

        logits, weights, preds = model(x, seqlens)
        logits = logits.squeeze()
        loss = criterion(logits, y)

        loss.backward()
        optimizer.step()

        with torch.no_grad():
            Preds.extend(preds.cpu().numpy())
            Labels.extend(y.cpu().numpy())

        if idx == 0:
            print("\n---------- Epoch {} ----------\n".format(epoch_num + 1))
            words = [data.TEXT.vocab.itos[idx] for idx in x[0]]
            print("Words:", words)
            print("Indexes:", x.cpu().numpy()[0])
            print("Label:", y[0].item())

        if idx % 10 == 0:
            print("Step: {}, Loss: {}".format(idx, loss.item()))

        train_loss += loss.item()

    train_results = calculate_results(Preds, Labels)

    epoch_loss = train_loss / len(train_iter)
    epoch_results = [x / len(train_iter) for x in train_results]

    return epoch_loss, epoch_results


def evaluate(model, eval_iter, criterion):
    """
    Evaluates the model after training.
    """
    eval_loss = 0
    y_pred, y_true = [], []

    model.eval()

    with torch.no_grad():
        for batch in eval_iter:
            x, seqlens = batch.text

            y = batch.label - 1

            logits, weights, preds = model(x, seqlens)
            logits = logits.squeeze()
            loss = criterion(logits, y)

            y_pred.extend(preds.cpu().numpy())
            y_true.extend(y.cpu().numpy())

            eval_loss += loss.item()

    epoch_loss = eval_loss / len(eval_iter)
    epoch_results = calculate_results(y_pred, y_true)

    return epoch_loss, epoch_results


def calculate_results(y_pred, y_true):
    """
    Obtains accuracy, precision, recall and F score for a training epoch,
    given an array of ground-truth values and an array of predictions.
    """
    y_pred = np.array(y_pred).squeeze()
    y_true = np.array(y_true)
    tp = np.sum(np.logical_and(y_pred == 1, y_true == 1))
    fp = np.sum(np.logical_and(y_pred == 1, y_true == 0))
    tn = np.sum(np.logical_and(y_pred == 0, y_true == 0))
    fn = np.sum(np.logical_and(y_pred == 0, y_true == 1))

    accuracy = (tp + tn) / (tp + tn + fp + fn)
    precision = tp / (tp + fp)
    recall = tp / (tp + fn)
    f1 = 2 * (precision * recall) / (precision + recall)
    results = [accuracy, precision, recall, f1]

    return results


def build_model(data):
    """
    Builds the neural model architecture for use in training or testing.
    """
    fasttext = data.TEXT.vocab.vectors

    vocab_size = len(data.TEXT.vocab)
    embed_d = fasttext.shape[1]
    hidden_d = embed_d // 2
    context_d = 150
    if s.args.dataset == 'imdb':
        output_d = 1
    else:
        output_d = 3

    pad_idx = data.TEXT.vocab.stoi[data.TEXT.pad_token]
    unk_idx = data.TEXT.vocab.stoi[data.TEXT.unk_token]

    model = Network(vocab_size, embed_d, hidden_d, output_d, context_d,
                    s.args.dropout, pad_idx)
    model.to(s.device)

    model.embedding.weight.data.copy_(fasttext)
    model.embedding.weight.data[unk_idx] = torch.zeros(embed_d)
    model.embedding.weight.data[pad_idx] = torch.zeros(embed_d)

    return model


def main(data, model, train_iter, valid_iter):
    torch.manual_seed(s.args.seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    optimizer = optim.Adam(model.parameters(), lr=s.args.lr,
                           weight_decay=s.args.weight_decay)

    if s.args.dataset == 'imdb':
        criterion = nn.BCEWithLogitsLoss()
            # Binary cross entropy with sigmoid
    else:
        criterion = nn.CrossEntropyLoss()
    criterion.to(s.device)

    best_valid_loss = np.inf

    if not os.path.exists(s.logdir):
        os.makedirs(s.logdir)

    for epoch in range(s.args.n_epochs):
        train_loss, train_results = train_mod(model, train_iter, optimizer,
                                              criterion, epoch)

        valid_loss, valid_results = evaluate(model, valid_iter,
                                             criterion)
        print("\n----- Results at Epoch {} -----\n".format(epoch + 1))
        print("Validation loss: {:.3f}".format(valid_loss))
        print("Accuracy: {:.3f}".format(valid_results[0]))
        print("Precision: {:.3f}".format(valid_results[1]))
        print("Recall: {:.3f}".format(valid_results[2]))
        print("F1: {:.3f}".format(valid_results[3]))

        best_model_path = os.path.join(s.logdir, str(epoch).zfill(2))
        if valid_loss < best_valid_loss:
            best_valid_loss = valid_loss
            torch.save(model.state_dict(),
                       "{}.pt".format(best_model_path))

    return model, best_model_path


def test(data, model, test_iter):
    """
    Obtains the loss and overall results on the test set.
    """
    vocab_size = len(data.TEXT.vocab)
    model.load_state_dict(torch.load('{}'.format(s.args.logdir)))

    if s.args.dataset == 'imdb':
        criterion = nn.BCEWithLogitsLoss()
    else:
        criterion = nn.CrossEntropyLoss()
    criterion.to(s.device)

    test_loss, test_results = evaluate(model, test_iter, criterion)
    print("\nTest loss: {:.3f}".format(test_loss))
    print("Accuracy: {:.3f}".format(test_results[0]))
    print("Precision: {:.3f}".format(test_results[1]))
    print("Recall: {:.3f}".format(test_results[2]))
    print("F1: {:.3f}".format(test_results[3]))


if __name__ == "__main__":
    data = DataPreparer(s.args.dataset)
    train_iter, valid_iter, test_iter = data.prepare_data()

    model = build_model(data)

    if s.args.train:
        trained_model, best_model_path = main(data, model, train_iter,
                                              valid_iter)
    else:
        test(data, model, test_iter)
