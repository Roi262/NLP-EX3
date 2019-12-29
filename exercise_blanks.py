import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
import os

from decimal import Decimal
from torch.autograd import Variable
from torch.utils.data import DataLoader, TensorDataset, Dataset
import operator
import data_loader
import pickle
import tqdm
from tqdm import tqdm
from matplotlib import pyplot as plt
from operator import add


# print(torch.__version__)


# Pytorch TA: https://colab.research.google.com/drive/18JCEnRsTmd_eSEkJmuoYGy0m0iOv7GgM

# ------------------------------------------- Constants ----------------------------------------

SEQ_LEN = 52
W2V_EMBEDDING_DIM = 300
LSTM_BIDIRECTIONAL = True

ONEHOT_AVERAGE = "onehot_average"
W2V_AVERAGE = "w2v_average"
W2V_SEQUENCE = "w2v_sequence"

TRAIN = "train"
VAL = "val"
TEST = "test"

NEG, POS, NEUTRAL, NEG_THRESH, POS_THRESH = 0., 1., -1., 0.4, 0.6

# ------------------------------------------ Helper methods and classes --------------------------


def get_available_device():
    """
    Allows training on GPU if available. Can help with running things faster when a GPU with cuda is
    available but not a most...
    Given a device, one can use module.to(device)
    and criterion.to(device) so that all the computations will be done on the GPU.
    """
    return torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def save_pickle(obj, path):
    with open(path, "wb") as f:
        pickle.dump(obj, f)


def load_pickle(path):
    with open(path, "rb") as f:
        return pickle.load(f)


def save_model(model, path, epoch, optimizer):
    """
    Utility function for saving checkpoint of a model, so training or evaluation can be executed later on.
    :param model: torch module representing the model
    :param optimizer: torch optimizer used for training the module
    :param path: path to save the checkpoint into
    """
    torch.save({
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict()}, path)


def load(model, path, optimizer):
    """
    Loads the state (weights, paramters...) of a model which was saved with save_model
    :param model: should be the same model as the one which was saved in the path
    :param path: path to the saved checkpoint
    :param optimizer: should be the same optimizer as the one which was saved in the path
    """
    checkpoint = torch.load(path)
    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    epoch = checkpoint['epoch']
    return model, optimizer, epoch


# ------------------------------------------ Data utilities ----------------------------------------

def load_word2vec():
    """ Load Word2Vec Vectors
        Return:
            wv_from_bin: All 3 million embeddings, each lengh 300
    """
    import gensim.downloader as api
    # wv_from_bin = create_or_load_slim_w2v
    wv_from_bin = api.load("word2vec-google-news-300")
    vocab = list(wv_from_bin.vocab.keys())
    print(wv_from_bin.vocab[vocab[0]])
    print("Loaded vocab size %i" % len(vocab))
    return wv_from_bin


def create_or_load_slim_w2v(words_list, cache_w2v=False):
    """
    returns word2vec dict only for words which appear in the dataset.
    :param words_list: list of words to use for the w2v dict
    :param cache_w2v: whether to save locally the small w2v dictionary
    :return: dictionary which maps the known words to their vectors
    """
    w2v_path = "w2v_dict.pkl"
    if not os.path.exists(w2v_path):
        full_w2v = load_word2vec()
        w2v_emb_dict = {k: full_w2v[k] for k in words_list if k in full_w2v}
        if cache_w2v:
            save_pickle(w2v_emb_dict, w2v_path)
    else:
        w2v_emb_dict = load_pickle(w2v_path)
    return w2v_emb_dict


def get_w2v_average(sent, word_to_vec, embedding_dim):
    """
    This method gets a sentence and returns the average word embedding of the words that make up
    the sentence.
    :param sent: the sentence object
    :param word_to_vec: a dictionary mapping words to their vector embeddings
    :param embedding_dim: the dimension of the word embedding vectors
    :return The average embedding vector as numpy ndarray.
    """
    avg = np.ndarray(embedding_dim)
    for word in sent.text:
        embedding = word_to_vec[word]
        avg += embedding
    avg = avg/len(sent.text)
    return avg


def get_one_hot(size, ind):
    """
    this method returns a one-hot vector of the given size, where the 1 is placed in the ind entry.
    :param size: the size of the vector
    :param ind: the entry index to turn to 1
    :return: numpy ndarray which represents the one-hot vector
    """
    array = np.zeros(size)
    array[ind] = 1
    return array


def average_one_hots(sent, word_to_ind):
    """
    this method gets a sentence, and a mapping between words to indices, and returns the average
    one-hot embedding of the tokens in the sentence.
    :param sent: a sentence object.
    :param word_to_ind: a mapping between words to indices
    :return:
    """
    avg = np.zeros(len(word_to_ind))
    for word in sent.text:
        word_index = word_to_ind[word]
        avg[word_index] += 1
    avg = avg/len(sent.text)
    return avg


def get_word_to_ind(words_list):
    """
    this function gets a list of words, and returns a mapping between
    words to their index.
    :param words_list: a list of words
    :return: the dictionary mapping words to the index
    """
    word_to_ind = {}
    # words_set = set(words_list)
    # TODO do we regard upper/lower case letters?
    for i in range(len(words_list)):
        word_to_ind[words_list[i]] = i
    return word_to_ind


def sentence_to_embedding(sent, word_to_vec, seq_len, embedding_dim=300):
    """
    this method gets a sentence and a word to vector mapping, and returns a list containing the
    words embeddings of the tokens in the sentence.
    :param sent: a sentence object
    :param word_to_vec: a word to vector mapping.
    :param seq_len: the fixed length for which the sentence will be mapped to.
    :param embedding_dim: the dimension of the w2v embedding
    :return: numpy ndarray of shape (seq_len, embedding_dim) with the representation of the sentence
    """
    embeddings = np.ndarray(shape=(seq_len, embedding_dim))
    for i, token in enumerate(sent):
        word_embedding = word_to_vec[token]
        embeddings[i] = word_embedding
    return embeddings


class OnlineDataset(Dataset):
    """
    A pytorch dataset which generates model inputs on the fly from sentences of SentimentTreeBank
    """

    def __init__(self, sent_data, sent_func, sent_func_kwargs):
        """
        :param sent_data: list of sentences from SentimentTreeBank
        :param sent_func: Function which converts a sentence to an input datapoint
        :param sent_func_kwargs: fixed keyword arguments for the state_func
        """
        self.data = sent_data
        self.sent_func = sent_func
        self.sent_func_kwargs = sent_func_kwargs

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        sent = self.data[idx]
        sent_emb = self.sent_func(sent, **self.sent_func_kwargs)
        sent_label = sent.sentiment_class
        return sent_emb, sent_label


class DataManager():
    """
    Utility class for handling all data management task. Can be used to get iterators for training and
    evaluation.
    """

    def __init__(self, data_type=ONEHOT_AVERAGE, use_sub_phrases=True, dataset_path="stanfordSentimentTreebank", batch_size=50,
                 embedding_dim=None):
        """
        builds the data manager used for training and evaluation.
        :param data_type: one of ONEHOT_AVERAGE, W2V_AVERAGE and W2V_SEQUENCE
        :param use_sub_phrases: if true, training data will include all sub-phrases plus the full sentences
        :param dataset_path: path to the dataset directory
        :param batch_size: number of examples per batch
        :param embedding_dim: relevant only for the W2V data types.
        """

        # load the dataset
        self.sentiment_dataset = data_loader.SentimentTreeBank(
            dataset_path, split_words=True)
        # map data splits to sentences lists
        self.sentences = {}
        if use_sub_phrases:
            self.sentences[TRAIN] = self.sentiment_dataset.get_train_set_phrases()
        else:
            self.sentences[TRAIN] = self.sentiment_dataset.get_train_set()

        self.sentences[VAL] = self.sentiment_dataset.get_validation_set()
        self.sentences[TEST] = self.sentiment_dataset.get_test_set()

        # map data splits to sentence input preperation functions
        words_list = list(self.sentiment_dataset.get_word_counts().keys())
        if data_type == ONEHOT_AVERAGE:
            self.sent_func = average_one_hots
            self.sent_func_kwargs = {
                "word_to_ind": get_word_to_ind(words_list)}
        elif data_type == W2V_SEQUENCE:
            self.sent_func = sentence_to_embedding

            self.sent_func_kwargs = {"seq_len": SEQ_LEN,
                                     "word_to_vec": create_or_load_slim_w2v(words_list, True),
                                     "embedding_dim": embedding_dim
                                     }
        elif data_type == W2V_AVERAGE:
            self.sent_func = get_w2v_average
            words_list = list(self.sentiment_dataset.get_word_counts().keys())
            self.sent_func_kwargs = {"word_to_vec": create_or_load_slim_w2v(words_list, True),
                                     "embedding_dim": embedding_dim
                                     }
        else:
            raise ValueError("invalid data_type: {}".format(data_type))
        # map data splits to torch datasets and iterators
        self.torch_datasets = {k: OnlineDataset(sentences, self.sent_func, self.sent_func_kwargs) for
                               k, sentences in self.sentences.items()}
        self.torch_iterators = {k: DataLoader(dataset, batch_size=batch_size, shuffle=k == TRAIN)
                                for k, dataset in self.torch_datasets.items()}

    def get_torch_iterator(self, data_subset=TRAIN):
        """
        :param data_subset: one of TRAIN VAL and TEST
        :return: torch batches iterator for this part of the datset
        """
        return self.torch_iterators[data_subset]

    def get_labels(self, data_subset=TRAIN):
        """
        :param data_subset: one of TRAIN VAL and TEST
        :return: numpy array with the labels of the requested part of the datset in the same order of the
        examples.
        """
        return np.array([sent.sentiment_class for sent in self.sentences[data_subset]])

    def get_input_shape(self):
        """
        :return: the shape of a single example from this dataset (only of x, ignoring y the label).
        """
        return self.torch_datasets[TRAIN][0][0].shape


# ------------------------------------ Models ----------------------------------------------------

class LSTM(nn.Module):
    """
    An LSTM for sentiment analysis with architecture as described in the exercise description.
    """

    def __init__(self, embedding_dim, hidden_dim, n_layers, drop_prob=.5):
        # self.dropout = nn.Dropout(drop_prob)
        self.lstm = nn.LSTM(input_size=embedding_dim, hidden_size=hidden_dim,
                            num_layers=n_layers, dropout=drop_prob, bidirectional=LSTM_BIDIRECTIONAL)

        return

    def forward(self, text):
        return

    def predict(self, text):
        return


class LSTM(nn.Module):
    def __init__(self, vocab_size, output_size, embedding_dim, hidden_dim, n_layers, drop_prob=0.5):
        super(LSTM, self).__init__()
        self.output_size = output_size
        self.n_layers = n_layers
        self.hidden_dim = hidden_dim

        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.lstm = nn.LSTM(embedding_dim, hidden_dim,
                            n_layers, dropout=drop_prob, batch_first=True)
        self.dropout = nn.Dropout(drop_prob)
        self.fc = nn.Linear(hidden_dim, output_size)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x, hidden):
        batch_size = x.size(0)
        x = x.long()
        embeds = self.embedding(x)
        lstm_out, hidden = self.lstm(embeds, hidden)
        lstm_out = lstm_out.contiguous().view(-1, self.hidden_dim)

        out = self.dropout(lstm_out)
        out = self.fc(out)
        out = self.sigmoid(out)

        out = out.view(batch_size, -1)
        out = out[:, -1]
        return out, hidden

    def init_hidden(self, batch_size):
        weight = next(self.parameters()).data
        hidden = (weight.new(self.n_layers, batch_size, self.hidden_dim).zero_().to(device),
                  weight.new(self.n_layers, batch_size, self.hidden_dim).zero_().to(device))
        return hidden


class LogLinear(nn.Module):
    """
    general class for the log-linear models for sentiment analysis.
    """

    def __init__(self, embedding_dim):
        super().__init__()  # Important to call Modules constructor!!
        self.linear = nn.Linear(in_features=embedding_dim, out_features=1)
        return

    def forward(self, x):
        """
        Arguments:
            x {[type]} -- the average one-hot embedding of the words in the sentence

        Returns:
            [type] -- [description]
        """
        return self.linear(x.float())

    def predict(self, x):
        # x = self.forward(x) #Removed since we dont want to forward twice.
        return torch.sigmoid(x)


# ------------------------- training functions -------------


def binary_accuracy(preds, y):
    """
    This method returns tha accuracy of the predictions, relative to the labels.
    You can choose whether to use numpy arrays or tensors here.
    :param preds: a vector of predictions
    :param y: a vector of true labels
    :return: scalar value - (<number of accurate predictions> / <number of examples>)
    """
    vec = (preds == y)
    return float(sum(vec))/float(len(vec))


def round_pred(pred_values):
    """Rounds the prediction to NEG, POS, or NEUTRAL and returns that value
    Arguments:
        pred {float} -- one prediction
    Returns:
        [float] -- rounded prediction
    """
    # rounded_predictions = [0] * pred_values.shape[0]
    # for i, val in enumerate(pred_values):
    #     if val <= NEG_THRESH:
    #         rounded_predictions[i] = NEG
    #     elif val >= POS_THRESH:
    #         rounded_predictions[i] = POS
    #     else:
    #         rounded_predictions[i] = NEUTRAL
    rounded_predictions = pred_values > 0.5

    return torch.tensor(rounded_predictions).type(torch.int)


def train_epoch(model, data_iterator, optimizer, criterion):
    """
    This method operates one epoch (pass over the whole train set) of training of the given model,
    and returns the accuracy and loss for this epoch
    :param model: the model we're currently training
    :param data_iterator: an iterator, iterating over the training data for the model.
    :param optimizer: the optimizer object for the training process.
    :param criterion: the criterion object for the training process.
    """
    avg_loss = []
    y_predictions = []
    avg_acc = []
    # loop over the dataset
    for x, y_label in tqdm(data_iterator):
        x = x.float()
        y_label = y_label.reshape(len(y_label), 1).double()
        y_forward = model(x)
        #############################
        y_p = model.predict(y_forward).double()

        y_predictions.append(y_p)
        # compute the loss
        loss = criterion(y_forward, y_label)  # CRITERION USES SIGMOID
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()  # nullify gradients
        avg_loss.append(loss.item())
        avg_acc.append(binary_accuracy(round_pred(y_p), y_label))

    epoch_acc = np.mean(avg_acc)  # Not efficient but good for control.
    epoch_loss = np.mean(avg_loss)
    return epoch_loss, epoch_acc


def evaluate(model, data_iterator, criterion):
    """
    evaluate the model performance on the given data
    :param model: one of our models..
    :param data_iterator: torch data iterator for the relevant subset
    :param criterion: the loss criterion used for evaluation
    :return: tuple of (average loss over all examples, average accuracy over all examples)
    """
    model.evaluate()
    avg_loss = []
    y_predictions = []
    avg_acc = []
    # loop over the dataset
    for x, y_label in tqdm(data_iterator):
        x = x.float()
        y_label = y_label.reshape(len(y_label), 1).double()
        y_forward = model(x)
        #############################
        y_p = model.predict(y_forward).double()

        y_predictions.append(y_p)
        # compute the loss
        loss = criterion(y_p, y_label)
        loss.backward()
        avg_loss.append(loss.item())
        avg_acc.append(binary_accuracy(round_pred(y_p), y_label))

    epoch_acc = np.mean(avg_acc)  # Not efficient but good for control.
    epoch_loss = np.mean(avg_loss)
    return epoch_loss, epoch_acc


def get_predictions_for_data(model, data_iter):
    """
    This function should iterate over all batches of examples from data_iter and return all of the models
    predictions as a numpy ndarray or torch tensor (or list if you prefer). the prediction should be in the
    same order of the examples returned by data_iter.
    :param model: one of the models you implemented in the exercise
    :param data_iter: torch iterator as given by the DataManager
    :return:
    """
    predictions = np.ndarray(len(data_iter))
    for i, example in enumerate(data_iter):
        predictions[i] = model.predict(example)
    return predictions


def train_model(model, data_manager, n_epochs, lr, weight_decay=0.):
    """
    Runs the full training procedure for the given model. The optimization should be done using the Adam
    optimizer with all parameters but learning rate and weight decay set to default.
    :param model: module of one of the models implemented in the exercise
    :param data_manager: the DataManager object
    :param n_epochs: number of times to go over the whole training set
    :param lr: learning rate to be used for optimization
    :param weight_decay: parameter for l2 regularization
    """
    model.train()  # This is here because of the inverse function in evaluate.
    optimizer = optim.Adam(params=model.parameters(),
                           lr=lr, weight_decay=weight_decay)
    criterion = nn.BCEWithLogitsLoss()
    train_acc_arr = []
    train_loss_arr = []

    val_acc_arr = []
    val_loss_arr = []
    iterator = data_manager.get_torch_iterator(data_subset=TRAIN)
    for epoch in (range(n_epochs)):
        avg_train_loss, avg_train_acc = train_epoch(
            model=model, data_iterator=iterator, optimizer=optimizer, criterion=criterion)
        train_acc_arr.append(avg_train_acc)
        train_loss_arr.append(avg_train_loss)

        # Validation        TODO is the validation in the same epoch loop or another?
        avg_val_loss, avg_val_acc = train_epoch(
            model, data_manager.get_torch_iterator(data_subset=VAL), optimizer, criterion)
        val_acc_arr.append(avg_val_acc)
        val_loss_arr.append(avg_val_loss)

    return train_acc_arr, train_loss_arr, val_acc_arr, val_loss_arr


def train_log_linear_with_one_hot(lr, n_epochs, weight_decay):
    """
    """
    # get data
    size = 64
    data_manager = DataManager(batch_size=size)
    # test_iterator = DataManager(batch_size=size).get_torch_iterator(data_subset=TEST)

    embedding_dimension = len(data_manager.sentiment_dataset.get_word_counts())

    # number of distinct words in the corpus
    log_linear_model = LogLinear(embedding_dim=embedding_dimension)
    train_acc, train_loss, val_acc, val_loss = train_model(model=log_linear_model, data_manager=data_manager, n_epochs=n_epochs,
                                                           lr=lr, weight_decay=weight_decay)
    # print("EVALUATE")
    # test_acc, test_loss = evaluate(log_linear_model, test_iterator, nn.BCEWithLogitsLoss())
    return train_acc, train_loss, val_acc, val_loss


def train_log_linear_with_w2v(lr, n_epochs, weight_decay):
    """
    """
    # get data
    size = 64
    data_manager = DataManager(
        batch_size=size, embedding_dim=W2V_EMBEDDING_DIM, data_type=W2V_AVERAGE)
    # test_iterator = DataManager(batch_size=size, embedding_dim=W2V_EMBEDDING_DIM).get_torch_iterator(data_subset=TEST)

    # embedding_dimension = len(data_manager.sentiment_dataset.get_word_counts())

    # number of distinct words in the corpus
    log_linear_w2v = LogLinear(embedding_dim=W2V_EMBEDDING_DIM)
    train_acc, train_loss, val_acc, val_loss = train_model(model=log_linear_w2v, data_manager=data_manager, n_epochs=n_epochs,
                                                           lr=lr, weight_decay=weight_decay)
    # print("EVALUATE")
    # test_acc, test_loss = evaluate(log_linear_model, test_iterator, nn.BCEWithLogitsLoss())
    return train_acc, train_loss, val_acc, val_loss


def train_lstm_with_w2v():
    """
    Here comes your code for training and evaluation of the LSTM model.
    """
    return


def plot_graphs(name_of_model, train_acc, train_loss, val_acc, val_loss, n_epochs, w_decay, lr, Q):
    """
    Plot Accuracy and Loss graphs.
    :param name_of_model:
    :param n_epochs: Number of epochs
    :param w_decay: weight decay
    :param lr: Learning rate
    All of the followings are arrays, with dimention of n_epochs.
    :param train_acc
    :param train_loss
    :param val_acc
    :param val_loss
    """

    dir_path = ("plots") + os.sep
    w_decimal = str(w_decay).replace('.', '')
    epoch_numbers = list(range(n_epochs))
    epoch_numbers = list(map(add, epoch_numbers, [1]*n_epochs))
    print(train_acc)

    plt.title(name_of_model + " Model Accuracy\n"
              "Decay weight="+str(w_decay)+", Learning "
              "rate="+str(lr))
    plt.plot(epoch_numbers, train_acc, label="Train Accuracy")
    plt.plot(epoch_numbers, val_acc, label="Validation Accuracy")
    plt.xticks(epoch_numbers)
    plt.xlabel("Epoch number")
    plt.ylabel("Accuracy rates")
    plt.legend()
    plt.grid()
    plt.savefig(dir_path + name_of_model + "_acc_w=" + str(w_decimal))
    plt.clf()  # clears the plot
    # plt.show()

    plt.title(name_of_model + " Model Loss\n"
              "Decay weight="+str(w_decay)+", Learning rate="+str(lr)+" "+Q)
    plt.plot(epoch_numbers, train_loss, label="Train Loss")
    plt.plot(epoch_numbers, val_loss, label="Validation Loss")
    plt.xticks(epoch_numbers)
    plt.xlabel("Epoch number")
    plt.ylabel("Loss rates")
    plt.legend()
    plt.grid()
    plt.savefig(dir_path + name_of_model + "_loss_w=" + str(w_decimal)+" "+Q)
    plt.clf()  # clears the plot
    # plt.show()


def Q1(lr, weights_array, n_epochs):
    """Plots graphs for accuracy and loss for each w_decay."""
    for w_dec in weights_array:
        print(str(w_dec)+" Q1")
        train_acc_arr, train_loss_arr, val_acc_arr, val_loss_arr = train_log_linear_with_one_hot(lr=lr,
                                                                                                 weight_decay=w_dec,
                                                                                                 n_epochs=n_epochs)

        plot_graphs("Log Linear with ONEHOT", train_acc_arr, train_loss_arr, val_acc_arr, val_loss_arr, n_epochs, w_dec,
                    lr, "Q2")


def Q2(lr, weights_array, n_epochs):
    for w_dec in weights_array:
        print(str(w_dec) + " Q2")
        train_acc_arr, train_loss_arr, val_acc_arr, val_loss_arr = train_log_linear_with_w2v(
            lr=lr, weight_decay=w_dec, n_epochs=n_epochs)
        plot_graphs("Log Linear with W2V", train_acc_arr, train_loss_arr, val_acc_arr, val_loss_arr, n_epochs, w_dec, lr,
                    "Q2")


def main():
    weights_array = [0, 0.0001, 0.001]
    n_epochs = 3
    lr = 0.0001
    Q2(lr, weights_array, n_epochs)

    # for wd in weight_decays:
    #     train_log_linear_with_one_hot(lr=lr1, weight_decay=wd, n_epochs=20)
    # train_log_linear_with_w2v()
    # train_lstm_with_w2v()


if __name__ == '__main__':
    main()
