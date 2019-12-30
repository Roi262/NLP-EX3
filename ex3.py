import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
import os
from torch.utils.data import DataLoader, TensorDataset, Dataset
import operator
import data_loader
import pickle
import tqdm
import matplotlib.pyplot as plt

# ------------------------------------------- Constants ----------------------------------------

# SEQ_LEN = 25
SEQ_LEN = 52
W2V_EMBEDDING_DIM = 300

ONEHOT_AVERAGE = "onehot_average"
W2V_AVERAGE = "w2v_average"
W2V_SEQUENCE = "w2v_sequence"

TRAIN = "train"
VAL = "val"
TEST = "test"

POS_THRESHOLD = 0.6
NEG_THRESHOLD = 0.4

PRED_THRESHOLD = 0.5

DEFAULT_BATCH_SIZE = 64

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

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
    This method gets a sentence and returns the average word embedding of the words consisting
    the sentence.
    :param sent: the sentence object
    :param word_to_vec: a dictionary mapping words to their vector embeddings
    :param embedding_dim: the dimension of the word embedding vectors
    :return The average embedding vector as numpy ndarray.
    """
    sum = np.zeros(embedding_dim)
    text = sent.text
    count = 0
    for word in text:
        if word in word_to_vec:
            count += 1
            sum += word_to_vec[word]

    if count == 0:
        return sum

    return sum / count


def get_one_hot(size, ind):
    """
    this method returns a one-hot vector of the given size, where the 1 is placed in the ind entry.
    :param size: the size of the vector
    :param ind: the entry index to turn to 1
    :return: numpy ndarray which represents the one-hot vector
    """
    one_hot = np.zeros(size)
    one_hot[ind] = 1
    return one_hot


def average_one_hots(sent, word_to_ind):
    """
    this method gets a sentence, and a mapping between words to indices, and returns the average
    one-hot embedding of the tokens in the sentence.
    :param sent: a sentence object.
    :param word_to_ind: a mapping between words to indices
    :return:
    """
    size = len(word_to_ind)
    sum_vec = np.zeros(size)
    text = sent.text

    if len(text) == 0:
        return sum_vec

    for word in text:
        sum_vec += get_one_hot(size, word_to_ind[word])

    sum_vec = sum_vec / len(text)
    return sum_vec


def get_word_to_ind(words_list):
    """
    this function gets a list of words, and returns a mapping between
    words to their index.
    :param words_list: a list of words
    :return: the dictionary mapping words to the index
    """
    words = list(set(words_list))
    words_dict = {}
    for i, word in enumerate(words):
        words_dict[word] = i

    return words_dict


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
    result = np.zeros([seq_len, embedding_dim])
    text = sent.text
    for i, word in enumerate(text[:seq_len]):
        if word in word_to_vec:
            result[i] = word_to_vec[word]

    return result


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
            self.sentences[TRAIN] = self.sentiment_dataset.get_set_phrases()
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
                                     "word_to_vec": create_or_load_slim_w2v(words_list, cache_w2v=True),
                                     "embedding_dim": embedding_dim
                                     }
        elif data_type == W2V_AVERAGE:
            self.sent_func = get_w2v_average
            words_list = list(self.sentiment_dataset.get_word_counts().keys())
            self.sent_func_kwargs = {"word_to_vec": create_or_load_slim_w2v(words_list, cache_w2v=True),
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

    def __init__(self, embedding_dim, hidden_dim, n_layers, dropout):
        super(LSTM, self).__init__()
        self.lstm = nn.LSTM(input_size=embedding_dim, hidden_size=hidden_dim,
                            num_layers=n_layers, bidirectional=True, batch_first=True)
        self.linear = torch.nn.Linear(self.lstm.hidden_size * 2, 1)
        self.drop_layer = torch.nn.Dropout(p=dropout)
        self.double()

    def forward(self, text):
        # text = text.cuda()
        out, _ = self.lstm.forward(text)
        last_out = out[:, -1, :]
        dropped_output = self.drop_layer(last_out)
        linear_output = self.linear(dropped_output)
        #linear_output = self.linear(last_out)

        return linear_output

    def predict(self, text):
        # assumes we are in eval mode
        self.eval()
        # output should be concatenation of two directions
        output = self.forward(text.double())
        sig = torch.nn.Sigmoid()
        self.train()
        return sig(output)


class LogLinear(nn.Module):
    """
    general class for the log-linear models for sentiment analysis.
    """

    def __init__(self, embedding_dim):
        super(LogLinear, self).__init__()
        self.linear = torch.nn.Linear(embedding_dim, 1)

    def forward(self, x):
        return self.linear(x.float())

    def predict(self, x):
        sig = torch.nn.Sigmoid()
        return sig(self.forward(x.float()))


# ------------------------- training functions -------------


def binary_accuracy(preds, y):
    """
    This method returns tha accuracy of the predictions, relative to the labels.
    You can choose whether to use numpy arrays or tensors here.
    :param preds: a vector of predictions
    :param y: a vector of true labels
    :return: scalar value - (<number of accurate predictions> / <number of examples>)
    """

    final_preds = (preds >= PRED_THRESHOLD).int()
    return torch.sum(final_preds == y) / float(len(final_preds))


def train_epoch(model, data_iterator, optimizer, criterion):
    """
    This method operates one epoch (pass over the whole train set) of training of the given model,
    and returns the accuracy and loss for this epoch
    :param model: the model we're currently training
    :param data_iterator: an iterator, iterating over the training data for the model.
    :param optimizer: the optimizer object for the training process.
    :param criterion: the criterion object for the training process.
    """
    running_loss = 0
    # model.train()
    for i, data in enumerate(data_iterator):
        input, label = data
        optimizer.zero_grad()
        output = model(input)
        output = output.view(output.shape[0])  # remove second dimension
        loss = criterion(output, label)
        loss.backward()
        optimizer.step()


def evaluate(model, data_iterator, criterion):
    """
    evaluate the model performance on the given data
    :param model: one of our models..
    :param data_iterator: torch data iterator for the relevant subset
    :param criterion: the loss criterion used for evaluation
    :return: tuple of (average loss over all examples, average accuracy over all examples)
    """
    # model.eval()
    running_loss = 0
    running_accuracy = 0
    i = 1
    count = 1
    for i, data in enumerate(data_iterator, 1):
        input, label = data
        output = model(input).detach()
        output = output.view(output.shape[0])  # remove second dimension
        prediction = model.predict(input).detach()
        prediction = prediction.view(
            prediction.shape[0])  # remove second dimension
        # multiplying, since last batch might have different size
        loss = criterion(output, label) * len(input)
        acc = binary_accuracy(prediction, label) * len(input)

        running_loss += loss
        running_accuracy += acc

        count += len(input)

    running_loss /= count
    running_accuracy /= count

    return (running_loss, running_accuracy)


def get_predictions_for_data(model, data_iter):
    """

    This function should iterate over all batches of examples from data_iter and return all of the models
    predictions as a numpy ndarray or torch tensor (or list if you prefer). the prediction should be in the
    same order of the examples returned by data_iter.
    :param model: one of the models you implemented in the exercise
    :param data_iter: torch iterator as given by the DataManager
    :return:
    """
    predictions = torch.zeros(0)
    for i, data in enumerate(data_iter):
        input, label = data
        prediction = model.predict(input)
        prediction = prediction.view(
            prediction.shape[0])  # remove second dimension
        predictions = torch.cat((predictions, prediction))

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
    optimizer = torch.optim.Adam(
        params=model.parameters(), lr=lr, weight_decay=weight_decay)
    criterion = torch.nn.modules.loss.BCEWithLogitsLoss()
    tloss = np.zeros(n_epochs)
    vloss = np.zeros(n_epochs)
    tacc = np.zeros(n_epochs)
    vacc = np.zeros(n_epochs)

    val_iterator = data_manager.get_torch_iterator(VAL)
    init_loss, init_acc = evaluate(model, val_iterator, criterion)
    print("loss", init_loss)
    print("acc", init_acc)
    for i in range(n_epochs):
        print("epoch: ", i)
        iterator = data_manager.get_torch_iterator()
        val_iterator = data_manager.get_torch_iterator(VAL)
        train_iterator = data_manager.get_torch_iterator(TRAIN)
        train_epoch(model, iterator, optimizer, criterion)
        tloss[i], tacc[i] = evaluate(model, train_iterator, criterion)
        vloss[i], vacc[i] = evaluate(model, val_iterator, criterion)

        print("loss", vloss[i])
        print("acc", vacc[i])
    return tloss, tacc, vloss, vacc


def train_model1(model, data_manager, n_epochs, lr, weight_decay=0.):
    """
    Runs the full training procedure for the given model. The optimization should be done using the Adam
    optimizer with all parameters but learning rate and weight decay set to default.
    :param model: module of one of the models implemented in the exercise
    :param data_manager: the DataManager object
    :param n_epochs: number of times to go over the whole training set
    :param lr: learning rate to be used for optimization
    :param weight_decay: parameter for l2 regularization
    """
    optimizer = torch.optim.Adam(
        params=model.parameters(), lr=lr, weight_decay=weight_decay)
    criterion = torch.nn.modules.loss.BCEWithLogitsLoss()
    tstloss = np.zeros(n_epochs)
    tstacc = np.zeros(n_epochs)
    tst1acc = np.zeros(n_epochs)
    tst2acc = np.zeros(n_epochs)

    tst_data = np.array(data_manager.sentences[TEST])
    labels = np.array(data_manager.get_labels(TEST))
    tst1_indexes = data_loader.get_negated_polarity_examples(tst_data)
    tst2_indexes = data_loader.get_rare_words_examples(
        tst_data, data_manager.sentiment_dataset)
    tst1_data = tst_data[tst1_indexes]
    tst2_data = tst_data[tst2_indexes]
    word2vec = load_pickle("w2v_dict.pkl")

    tst1_data = [sentence_to_embedding(
        sent, word2vec, SEQ_LEN) for sent in tst1_data]
    tst2_data = [sentence_to_embedding(
        sent, word2vec, SEQ_LEN) for sent in tst2_data]
    tst1_labels = labels[tst1_indexes]
    tst2_labels = labels[tst2_indexes]

    tst1_iterator = [[torch.FloatTensor([tst1_data[i]]).double().cuda(), torch.FloatTensor(
        [tst1_labels[i]]).double().cuda()] for i in range(len(tst1_indexes))]
    tst2_iterator = [[torch.FloatTensor([tst2_data[i]]).double().cuda(), torch.FloatTensor(
        [tst2_labels[i]]).double().cuda()] for i in range(len(tst2_indexes))]
    for i in range(n_epochs):
        print("epoch: ", i)
        iterator = data_manager.get_torch_iterator()
        tst_iterator = data_manager.get_torch_iterator(TEST)

        train_epoch(model, iterator, optimizer, criterion)
        tstloss[i], tstacc[i] = evaluate(model, tst_iterator, criterion)
        _, tst1acc[i] = evaluate(model, tst1_iterator, criterion)
        _, tst2acc[i] = evaluate(model, tst2_iterator, criterion)

        print("loss", tstloss[i])
        print("acc", tstacc[i])
    return tstloss, tstacc, tst1acc, tst2acc


def train_log_linear_with_one_hot():
    """
    Here comes your code for training and evaluation of the log linear model with one hot representation.
    """
    weight_decays = [0, 0.0001, 0.001]
    for weight_decay in weight_decays:
        data_manager = DataManager(
            data_type=ONEHOT_AVERAGE, batch_size=DEFAULT_BATCH_SIZE)
        dim = data_manager.get_input_shape()[0]
        model = LogLinear(dim)
        tloss, tacc, vloss, vacc = train_model(
            model, data_manager=data_manager, n_epochs=20, lr=0.01, weight_decay=weight_decay)
        plt.plot(np.arange(len(tacc)) + 1, tacc,
                 color="blue", label="training accuracy")
        plt.plot(np.arange(len(vacc)) + 1, vacc,
                 color="red", label="validation accuracy")
        plt.title(
            "One-hot average accuracy as a function of epochs, weight_decay=%s" % (weight_decay,))
        plt.xlabel("epochs")
        plt.ylabel("accuracy")
        plt.legend()
        plt.show()

        plt.plot(np.arange(len(tloss)) + 1, tloss,
                 color="blue", label="training loss")
        plt.plot(np.arange(len(vloss)) + 1, vloss,
                 color="red", label="validation loss")
        plt.title(
            "One-hot average loss as a function of epochs, weight_decay=%s" % (weight_decay,))
        plt.xlabel("epochs")
        plt.ylabel("loss")
        plt.legend()
        plt.show()
    return model


def train_log_linear_with_w2v():
    """
    Here comes your code for training and evaluation of the log linear model with word embeddings
    representation.
    """
    weight_decays = [0, 0.0001, 0.001]
    for weight_decay in weight_decays:
        data_manager = DataManager(
            data_type=W2V_AVERAGE, batch_size=DEFAULT_BATCH_SIZE, embedding_dim=W2V_EMBEDDING_DIM)
        dim = data_manager.get_input_shape()[0]
        model = LogLinear(dim)
        tloss, tacc, vloss, vacc = train_model(
            model, data_manager=data_manager, n_epochs=5, lr=0.01, weight_decay=weight_decay)
        plt.plot(np.arange(len(tacc)) + 1, tacc,
                 color="blue", label="training accuracy")
        plt.plot(np.arange(len(vacc)) + 1, vacc,
                 color="red", label="validation accuracy")
        plt.title("W2V average accuracy as a function of epochs, weight_decay=%s" % (
            weight_decay,))
        plt.xlabel("epochs")
        plt.ylabel("accuracy")
        plt.legend()
        plt.show()

        plt.plot(np.arange(len(tloss)) + 1, tloss,
                 color="blue", label="training loss")
        plt.plot(np.arange(len(vloss)) + 1, vloss,
                 color="red", label="validation loss")
        plt.title(
            "W2V average loss as a function of epochs, weight_decay=%s" % (weight_decay,))
        plt.xlabel("epochs")
        plt.ylabel("loss")
        plt.legend()
        plt.show()
    return model


def train_lstm_with_w2v():
    """
    Here comes your code for training and evaluation of the LSTM model.
    """
    epochs = 20
    data_manager = DataManager(
        data_type=W2V_SEQUENCE, batch_size=DEFAULT_BATCH_SIZE, embedding_dim=W2V_EMBEDDING_DIM)
    model = LSTM(embedding_dim=W2V_EMBEDDING_DIM,
                 hidden_dim=100, n_layers=1, dropout=0.5)
    # model.cuda()
    tloss, tacc, vloss, vacc = train_model(
        model, data_manager=data_manager, n_epochs=epochs, lr=0.01)
    plt.plot(np.arange(len(tacc)) + 1, tacc,
             color="blue", label="training accuracy")
    plt.plot(np.arange(len(vacc)) + 1, vacc,
             color="red", label="validation accuracy")
    plt.title("BiLSTM average accuracy as a function of epochs, dropout=0.5")
    plt.xlabel("epochs")
    plt.ylabel("accuracy")
    plt.legend()
    # plt.show()
    plt.savefig("BILSTM_accuracy_%s.png" % (epochs))
    plt.close()

    plt.plot(np.arange(len(tloss)) + 1, tloss,
             color="blue", label="training loss")
    plt.plot(np.arange(len(vloss)) + 1, vloss,
             color="red", label="validation loss")
    plt.title("BiLSTM average loss as a function of epochs, dropout=0.5")
    plt.xlabel("epochs")
    plt.ylabel("loss")
    plt.legend()
    # plt.show()
    plt.savefig("BILSTM_loss_%s.png" % (epochs))
    plt.close()
    return model


def get_lstm_results():
    """
    Here comes your code for training and evaluation of the LSTM model.
    """
    epochs = 20
    data_manager = DataManager(
        data_type=W2V_SEQUENCE, batch_size=DEFAULT_BATCH_SIZE, embedding_dim=W2V_EMBEDDING_DIM)
    model = LSTM(embedding_dim=W2V_EMBEDDING_DIM,
                 hidden_dim=100, n_layers=1, dropout=0.5)
    tstloss, tstacc, tst1acc, tst2acc = train_model1(
        model, data_manager=data_manager, n_epochs=epochs, lr=0.01)
    plt.plot(np.arange(len(tstacc)) + 1, tstacc, color="green")
    plt.title("BiLSTM test accuracy as a function of epochs, dropout=0.5")
    plt.xlabel("epochs")
    plt.ylabel("test accuracy")
    plt.savefig("BILSTM_test_accuracy_%s.png" % (epochs))
    plt.close()

    plt.plot(np.arange(len(tstloss)) + 1, tstloss, color="green")
    plt.title("BiLSTM test loss as a function of epochs, dropout=0.5")
    plt.xlabel("epochs")
    plt.ylabel("accuracy")
    plt.savefig("BILSTM_test_loss_%s.png" % (epochs))
    plt.close()

    plt.plot(np.arange(len(tst1acc)) + 1, tst1acc, color="green")
    plt.title("BiLSTM negated polarity accuracy as a function of epochs")
    plt.xlabel("epochs")
    plt.ylabel("accuracy")
    plt.savefig("BILSTM_test_accuracy_negpol_%s.png" % (epochs))
    plt.close()

    plt.plot(np.arange(len(tst2acc)) + 1, tst2acc, color="green")
    plt.title("BiLSTM rare words accuracy as a function of epochs")
    plt.xlabel("epochs")
    plt.ylabel("accuracy")
    plt.savefig("BILSTM_test_accuracy_rare_%s.png" % (epochs))
    plt.close()

    return model


if __name__ == '__main__':
    # torch.cuda.device(DEVICE)
    # torch.set_default_tensor_type(torch.cuda.FloatTensor)
    # train_log_linear_with_one_hot()
    # train_log_linear_with_w2v()
    train_lstm_with_w2v()
    # get_lstm_results()
