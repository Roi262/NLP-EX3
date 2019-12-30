import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.parameter import Parameter
import torch.optim as optim
import numpy as np
import os
from torch.utils.data import DataLoader, TensorDataset, Dataset
import operator
import data_loader
import pickle
from tqdm import tqdm
import matplotlib.pyplot as plt
import math

from data_loader import SentimentTreeBank

# -------------------------- Constants ----------------------------------------

SEQ_LEN = 52
W2V_EMBEDDING_DIM = 300

ONE_HOT_AVERAGE = "onehot_average"
W2V_AVERAGE = "w2v_average"
W2V_SEQUENCE = "w2v_sequence"

TRAIN = "train"
VAL = "val"
TEST = "test"


# --------------------- Helper methods and classes --------------------------

def get_available_device():
	"""
	Allows training on GPU if available. Can help with running things faster
	when a GPU with cuda is available but not a most...
	Given a device, one can use module.to(device)
	and criterion.to(device) so that all the computations will be done on the
	GPU.
	"""
	return torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def save_pickle(obj, path):
	with open(path, "wb") as f:
		pickle.dump(obj, f)


def load_pickle(path):
	with open(path, "rb") as f:
		return pickle.load(f)


def save_model(model, path, epoch, optimizer=None):
	"""
	Utility function for saving checkpoint of a model, so training or
	evaluation can be executed later on.
	:param model: torch module representing the model
	:param path: path to save the checkpoint into
	:param epoch: todo
	:param optimizer: torch optimizer used for training the module
	"""
	if optimizer:
		torch.save({'epoch': epoch, 'model_state_dict': model.state_dict(),
		            'optimizer_state_dict': optimizer.state_dict()}, path)
	else:
		torch.save({
			'epoch': epoch,
			'model_state_dict': model.state_dict()}, path)


def load(model, path, optimizer=None):
	"""
	Loads the state (weights, parameters...) of a model which was saved with
	save_model
	:param model: should be the same model as the one which was saved in the
	path
	:param path: path to the saved checkpoint
	:param optimizer: should be the same optimizer as the one which was saved
	in the path
	"""
	checkpoint = torch.load(path)
	model.load_state_dict(checkpoint['model_state_dict'])
	epoch = checkpoint['epoch']
	if optimizer:
		optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
	return model, optimizer, epoch


# ---------------------- Data utilities -------------------------------------

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


def create_or_load_slim_w2v(words_list, cache_w2v=True):
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
	This method gets a sentence and returns the average word embedding of the
	words consisting the sentence.
	:param sent: the sentence object
	:param word_to_vec: a dictionary mapping words to their vector embeddings
	:param embedding_dim: the dimension of the word embedding vectors
	:return The average embedding vector as numpy ndarray.
	"""
	w2v_average = np.zeros(embedding_dim)
	average_counter = 0
	for word in sent.text:
		if word in word_to_vec:
			w2v_average += word_to_vec[word]
			average_counter += 1
	if average_counter:
		return w2v_average / average_counter
	return w2v_average


def get_one_hot(size, ind):
	"""
	this method returns a one-hot vector of the given size, where the 1 is
	placed in the ind entry.
	:param size: the size of the vector
	:param ind: the entry index to turn to 1
	:return: numpy ndarray which represents the one-hot vector
	"""
	one_hot = np.zeros(size)
	one_hot[ind] = 1
	return one_hot


def average_one_hots(sent, word_to_ind):
	"""
	this method gets a sentence, and a mapping between words to indices, and
	returns the average one-hot embedding of the tokens in the sentence.
	:param sent: a sentence object.
	:param word_to_ind: a mapping between words to indices
	:return: one-hot embedding of the tokens in the sentence.
	"""
	one_hot_average = np.zeros(len(word_to_ind))

	for word in sent.text:
		# one_hot_average[word_to_ind[word.text[0]]] += 1
		one_hot_average[word_to_ind[word]] += 1
	returned = one_hot_average / len(sent.text)
	return returned


def get_word_to_ind(words_list):
	"""
	this function gets a list of words, and returns a mapping between
	words to their index.
	:param words_list: a list of words
	:return: the dictionary mapping words to the index
	"""
	word_idx_dict = {}
	for word in words_list:
		if word not in word_idx_dict:
			word_idx_dict[word] = len(word_idx_dict)
	# note that len is O(1) in pythons dicts
	return word_idx_dict


def sentence_to_embedding(sent, word_to_vec, seq_len, embedding_dim=300):
	"""
	this method gets a sentence and a word to vector mapping, and returns a
	list containing the words embeddings of the tokens in the sentence.
	:param sent: a sentence object
	:param word_to_vec: a word to vector mapping.
	:param seq_len: the fixed length for which the sentence will be mapped to.
	:param embedding_dim: the dimension of the w2v embedding
	:return: numpy ndarray of shape (seq_len, embedding_dim) with the
	representation of the sentence
	"""
	embedded_words = 0
	word_idx = 0
	return_array = np.zeros((seq_len, embedding_dim))
	sent_len = len(sent.text)
	while embedded_words < seq_len and word_idx < sent_len:
		word = sent.text[word_idx]
		if word in word_to_vec:
			return_array[embedded_words] = word_to_vec[word]
			embedded_words += 1
		word_idx += 1
	return return_array


class OnlineDataset(Dataset):
	"""
	A pytorch dataset which generates model inputs on the fly from sentences
	of SentimentTreeBank
	"""

	def __init__(self, sent_data, sent_func, sent_func_kwargs):
		"""
		:param sent_data: list of sentences from SentimentTreeBank
		:param sent_func: Function which converts a sentence to an input
		datapoint
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
	Utility class for handling all data management task. Can be used to get
	iterators for training and evaluation.
	"""

	def __init__(self, data_type=ONE_HOT_AVERAGE, use_sub_phrases=True,
	             dataset_path="stanfordSentimentTreebank", batch_size=50,
	             embedding_dim=None):
		"""
		builds the data manager used for training and evaluation.
		:param data_type: one of ONEHOT_AVERAGE, W2V_AVERAGE and W2V_SEQUENCE
		:param use_sub_phrases: if true, training data will include all
		sub-phrases plus the full sentences
		:param dataset_path: path to the dataset directory
		:param batch_size: number of examples per batch
		:param embedding_dim: relevant only for the W2V data types.
		"""

		# load the dataset
		self.sentiment_dataset = data_loader. \
			SentimentTreeBank(dataset_path, split_words=True)
		# map data splits to sentences lists
		self.sentences = {}
		if use_sub_phrases:
			self.sentences[
				TRAIN] = self.sentiment_dataset.get_train_set_phrases()
		else:
			self.sentences[TRAIN] = self.sentiment_dataset.get_train_set()

		self.sentences[VAL] = self.sentiment_dataset.get_validation_set()
		self.sentences[TEST] = self.sentiment_dataset.get_test_set()

		# map data splits to sentence input preperation functions
		words_list = list(self.sentiment_dataset.get_word_counts().keys())
		if data_type == ONE_HOT_AVERAGE:
			self.sent_func = average_one_hots
			self.sent_func_kwargs = {
				"word_to_ind": get_word_to_ind(words_list)}
		elif data_type == W2V_SEQUENCE:
			self.sent_func = sentence_to_embedding

			self.sent_func_kwargs = {"seq_len": SEQ_LEN,
			                         "word_to_vec": create_or_load_slim_w2v(
				                         words_list),
			                         "embedding_dim": embedding_dim
			                         }
		elif data_type == W2V_AVERAGE:
			self.sent_func = get_w2v_average
			words_list = list(self.sentiment_dataset.get_word_counts().keys())
			self.sent_func_kwargs = {
				"word_to_vec": create_or_load_slim_w2v(words_list),
				"embedding_dim": embedding_dim
			}
		else:
			raise ValueError("invalid data_type: {}".format(data_type))
		# map data splits to torch datasets and iterators
		self.torch_datasets = {
			k: OnlineDataset(sentences, self.sent_func, self.sent_func_kwargs)
			for k, sentences in tqdm(self.sentences.items())}
		self.torch_iterators = {k: DataLoader(dataset, batch_size=batch_size, shuffle=k == TRAIN)
		                        for k, dataset in self.torch_datasets.items()
		                        }

	def get_torch_iterator(self, data_subset=TRAIN):
		"""
		:param data_subset: one of TRAIN VAL and TEST
		:return: torch batches iterator for this part of the datset
		"""
		return self.torch_iterators[data_subset]

	def get_labels(self, data_subset=TRAIN):
		"""
		:param data_subset: one of TRAIN VAL and TEST
		:return: numpy array with the labels of the requested part of the
		datset in the same order of the
		examples.
		"""
		return np.array(
			[sent.sentiment_class for sent in self.sentences[data_subset]])

	def get_input_shape(self):
		"""
		:return: the shape of a single example from this dataset (only of x,
		ignoring y the label).
		"""
		return self.torch_datasets[TRAIN][0][0].shape


# ------------------------------------ Models ----------------------------------------------------

class LSTM(nn.Module):
	"""
	An LSTM for sentiment analysis with architecture as described in the
	exercise description.
	"""

	def __init__(self, embedding_dim, hidden_dim, n_layers, dropout):
		super().__init__()
		self.hidden_dim = hidden_dim
		self.embedding_dim = embedding_dim
		self.n_layers = n_layers
		self.lstm = nn.LSTM(input_size=embedding_dim, hidden_size=hidden_dim, num_layers=n_layers, bidirectional=True)
		self.dropout = nn.Dropout(p=dropout)
		# self.fc = nn.Linear((hidden_dim * 2), 1)
		self.fc = nn.Linear((hidden_dim), 1)

		return

	def forward(self, text):
		text = text.permute(1, 0, 2)
		out1, (h_n, c_n) = self.lstm(text)
		h_n = h_n[0, :, :] + h_n[1, :, :]
		mid = self.dropout(h_n)
		# out = self.fc(mid[:, -1, :])
		out = self.fc(mid)
		return out

	def predict(self, text):
		return torch.sigmoid(text)


class LogLinear(nn.Module):
	"""
	general class for the log-linear models for sentiment analysis.
	"""

	def __init__(self, embedding_dim):
		super().__init__()
		self.input_dim = embedding_dim
		self.output_dim = 1
		self.linear = nn.Linear(in_features=self.input_dim,
		                        out_features=self.output_dim)

	def __call__(self, *args, **kwargs):
		return self.forward(*args)

	def forward(self, x):
		return self.linear(x)

	def predict(self, x):
		return torch.sigmoid(x)


# ------------------------- training functions -------------


def binary_accuracy(preds, y):
	"""
	This method returns the accuracy of the predictions, relative to the labels.
	You can choose whether to use numpy arrays or tensors here.
	:param preds: a vector of predictions
	:param y: a vector of true labels
	:return: scalar value - (<number of accurate predictions> /
	<number of examples>)
	"""
	return np.mean(preds == y)


def train_epoch(model, data_iterator, optimizer, criterion):
	"""
	This method operates one epoch (pass over the whole train set) of training
	of the given model,
	and returns the accuracy and loss for this epoch
	:param model: the model we're currently training
	:param data_iterator: an iterator, iterating over the training data for
	the model.
	:param optimizer: the optimizer object for the training process.
	:param criterion: the criterion object for the training process.
	"""

	accumulated_accuracy = 0
	accumulated_loss = 0
	model.train()
	num_of_batches = np.ceil(data_iterator.sampler.num_samples / data_iterator.batch_size)
	for batch in tqdm(data_iterator):
		x = batch[0].float()
		y = batch[1].reshape(len(batch[1]), 1).double()

		optimizer.zero_grad()
		forward_out = model.forward(x)

		probs = model.predict(forward_out).double()
		loss = criterion(probs, y)
		loss.backward()
		optimizer.step()

		predictions = (probs > 0.5).int().numpy()
		this_round_acc = binary_accuracy(predictions, y.numpy())
		accumulated_accuracy += this_round_acc
		accumulated_loss += loss.item()
	return (accumulated_accuracy / num_of_batches, accumulated_loss / num_of_batches)


def evaluate(model, data_iterator, criterion):
	"""
	evaluate the model performance on the given data
	:param model: one of our models..
	:param data_iterator: torch data iterator for the relevant subset
	:param criterion: the loss criterion used for evaluation
	:return: tuple of (average loss over all examples, average accuracy over
	all examples)
	"""
	accumulated_accuracy = 0
	accumulated_loss = 0
	# num_of_batches = np.ceil(data_iterator.sampler.num_samples/data_iterator.batch_size)
	num_of_batches = 0
	model.eval()
	for batch in tqdm(data_iterator):
		x = batch[0].float()
		y = batch[1].reshape(len(batch[1]), 1).double()
		forward_out = model.forward(x)
		probs = model.predict(forward_out).double()
		predictions = (probs > 0.5).int().numpy()
		loss = criterion(probs, y)
		accumulated_accuracy += binary_accuracy(predictions, y.numpy())

		accumulated_loss += loss.item()
		num_of_batches += 1
	return (accumulated_accuracy / num_of_batches, accumulated_loss / num_of_batches)


def get_predictions_for_data(model, data_iter):
	"""
	This function should iterate over all batches of examples from data_iter
	and return all of the models predictions as a numpy ndarray or torch tensor (or list if you prefer).
	the prediction should be in the same order of the examples returned by data_iter.
	:param model: one of the models you implemented in the exercise
	:param data_iter: torch iterator as given by the DataManager
	:return:
	"""
	predictions = np.reshape(np.array([]), (0, 1))
	for batch in tqdm(data_iter):
		x = batch[0].float()
		forward_out = model.forward(x)
		probs = model.predict(forward_out).double()
		predictions = np.concatenate((predictions, (probs > 0.5).numpy()))
	return predictions


def train_model(model, data_manager, n_epochs, lr, weight_decay=0.):
	"""
	Runs the full training procedure for the given model. The optimization
	should be done using the Adam
	optimizer with all parameters but learning rate and weight decay set to
	default.
	:param model: module of one of the models implemented in the exercise
	:param data_manager: the DataManager object
	:param n_epochs: number of times to go over the whole training set
	:param lr: learning rate to be used for optimization
	:param weight_decay: parameter for l2 regularization
	"""
	optimizer = torch.optim.Adam(params=model.parameters(), lr=lr, weight_decay=weight_decay)
	train_iterator = data_manager.get_torch_iterator(data_subset=TRAIN)
	validation_iterator = data_manager.get_torch_iterator(data_subset=VAL)
	criterion = nn.BCEWithLogitsLoss()
	train_accuracy_list = []
	train_loss_list = []
	val_accuracy_list = []
	val_loss_list = []
	for i in tqdm(range(n_epochs)):
		accuracy, loss = train_epoch(model, train_iterator, optimizer, criterion)
		train_accuracy_list.append(accuracy)
		train_loss_list.append(loss)
		print("finished epoch {}".format(i + 1))
		val_accuracy, val_loss = evaluate(model, validation_iterator, criterion)
		val_accuracy_list.append(val_accuracy)
		val_loss_list.append(val_loss)
	return val_accuracy_list, val_loss_list, train_accuracy_list, train_loss_list


def plot_weight_decay_loss_accuracy(weight_decay_dict, model_name, epoch_num_list):
	save_path = "saved_figs/{0}/{1}_wd_{2}_plotting.png"
	if not os.path.isdir("saved_figs/{}".format(model_name)):
		os.mkdir("saved_figs/{}".format(model_name))
	for w in weight_decay_dict.keys():
		loss_fig, loss_ax = plt.subplots()
		loss_ax.plot(epoch_num_list, weight_decay_dict[w]["train_loss_list"], color="b", label="train")
		loss_ax.plot(epoch_num_list, weight_decay_dict[w]["val_loss_list"], color="r", label="validation")
		loss_ax.set(xlabel="epoch_num", ylabel="loss",
		            title="{}: loss plot as a function of epoch num,\nfor weight decay {}".format(model_name, w))
		loss_ax.legend()
		loss_ax.grid()
		loss_fig.savefig(save_path.format(model_name, "loss", w))
		plt.show()
		acc_fig, acc_ax = plt.subplots()
		acc_ax.plot(epoch_num_list, weight_decay_dict[w]["train_accuracy_list"], color="b", label="train")
		acc_ax.plot(epoch_num_list, weight_decay_dict[w]["val_accuracy_list"], color="r", label="validation")
		acc_ax.legend()
		acc_ax.set(xlabel="epoch_num", ylabel="accuracy",
		           title="{}: accuracy plot as a function of epoch num,\nfor weight decay {}".format(model_name, w))
		acc_ax.grid()
		acc_fig.savefig(save_path.format(model_name, "accuracy", w))
		plt.show()


def get_best_model_weight(weight_decay_dict):
	max_acc = 0
	max_acc_weight = 0
	for key, val in weight_decay_dict.items():
		acc = val["val_accuracy_list"][-1]
		if acc > max_acc:
			max_acc = acc
			max_acc_weight = key
	return max_acc_weight


def train_log_linear_with_one_hot(should_load=False):
	"""
	Here comes your code for training and evaluation of the log linear model
	with one hot representation.
	"""
	dataset = data_loader.SentimentTreeBank()
	model_name = "log-linear"
	pickle_path = "{}_pickled_weight_decay_dict.p".format(model_name)
	# if not os.path.isdir("saved_models/{}".format(model_name)):
	# 	os.mkdir("saved_models/{}".format(model_name))
	model_save_path = "saved_models/{0}/after_wd_{1}.pth"
	BATCH_SIZE = 64
	data_manager = DataManager(batch_size=BATCH_SIZE)
	LEARNING_RATE = 1e-3

	N_EPOCHS = 20
	if not should_load:
		weight_decay_dict = {0: {}, 0.0001: {}, 0.001: {}}
		for weight_decay in tqdm(weight_decay_dict.keys()):
			log_linear = LogLinear(embedding_dim=len(data_manager.sentiment_dataset.get_word_counts()))
			val_accuracy_list, val_loss_list, train_accuracy_list, train_loss_list = train_model(log_linear,
			                                                                                     data_manager, N_EPOCHS,
			                                                                                     LEARNING_RATE,
			                                                                                     weight_decay)
			weight_decay_dict[weight_decay]["val_loss_list"] = val_loss_list
			weight_decay_dict[weight_decay]["val_accuracy_list"] = val_accuracy_list
			weight_decay_dict[weight_decay]["train_accuracy_list"] = train_accuracy_list
			weight_decay_dict[weight_decay]["train_loss_list"] = train_loss_list
			save_model(log_linear, model_save_path.format(model_name, weight_decay), 0)
		save_pickle(weight_decay_dict, pickle_path)
	else:
		weight_decay_dict = load_pickle(pickle_path)
		log_linear = LogLinear(embedding_dim=len(data_manager.sentiment_dataset.get_word_counts()))
	epoch_num_list = np.arange(1, N_EPOCHS + 1)
	plot_weight_decay_loss_accuracy(weight_decay_dict, model_name, epoch_num_list)
	best_weight = get_best_model_weight(weight_decay_dict)
	model = load(log_linear, model_save_path.format(model_name, best_weight))[0]
	criterion = nn.BCEWithLogitsLoss()
	test_iterator = data_manager.get_torch_iterator(data_subset=TEST)
	test_acc, test_loss = evaluate(model, test_iterator, criterion)
	print("for model{}:\ntest accuracy is {}\ntest loss is {}".format(model_name, test_acc, test_loss))
	get_special_acc(model, dataset, test_iterator)
	return


def get_special_acc(model, dataset, test_iterator):
	sentences = test_iterator.dataset.data
	y = np.array([])
	for batch in test_iterator:
		y = np.concatenate((y, batch[1].numpy()))
	rare_words_idxs = data_loader.get_rare_words_examples(sentences, dataset)
	negated_polarity_idxs = data_loader.get_negated_polarity_examples(sentences)
	preds = get_predictions_for_data(model, test_iterator)
	rare_words_preds = preds[rare_words_idxs]
	negated_polarity_preds = preds[negated_polarity_idxs]
	rare_words_gt = y[rare_words_idxs]
	negated_polarity_gt = y[negated_polarity_idxs]
	rare_test_acc = binary_accuracy(rare_words_preds, rare_words_gt)
	negated_polarity_test_acc = binary_accuracy(negated_polarity_preds, negated_polarity_gt)
	print("rare words accuracy is {}\nnegated polarity acc is {}".format(rare_test_acc, negated_polarity_test_acc))


def train_log_linear_with_w2v(should_load=False):
	"""
	Here comes your code for training and evaluation of the log linear model
	with word embeddings
	representation.
	"""
	dataset = data_loader.SentimentTreeBank()
	model_name = "w2v"
	pickle_path = "{}_pickled_weight_decay_dict.p".format(model_name)
	if not os.path.isdir("saved_models/{}".format(model_name)):
		os.mkdir("saved_models/{}".format(model_name))
	model_save_path = "saved_models/{0}/after_wd_{1}.pth"
	BATCH_SIZE = 64
	data_manager = DataManager(data_type=W2V_AVERAGE, batch_size=BATCH_SIZE, embedding_dim=W2V_EMBEDDING_DIM)
	LEARNING_RATE = 1e-3

	N_EPOCHS = 20
	if not should_load:
		weight_decay_dict = {0: {}, 0.0001: {}, 0.001: {}}
		for weight_decay in tqdm(weight_decay_dict.keys()):
			w2v = LogLinear(embedding_dim=W2V_EMBEDDING_DIM)
			val_accuracy_list, val_loss_list, train_accuracy_list, train_loss_list = train_model(w2v, data_manager,
			                                                                                     N_EPOCHS,
			                                                                                     LEARNING_RATE,
			                                                                                     weight_decay)
			weight_decay_dict[weight_decay]["val_loss_list"] = val_loss_list
			weight_decay_dict[weight_decay]["val_accuracy_list"] = val_accuracy_list
			weight_decay_dict[weight_decay]["train_accuracy_list"] = train_accuracy_list
			weight_decay_dict[weight_decay]["train_loss_list"] = train_loss_list
			save_model(w2v, model_save_path.format(model_name, weight_decay), 0)
		save_pickle(weight_decay_dict, pickle_path)
	else:
		weight_decay_dict = load_pickle(pickle_path)
		w2v = LogLinear(embedding_dim=W2V_EMBEDDING_DIM)
	epoch_num_list = np.arange(1, N_EPOCHS + 1)
	plot_weight_decay_loss_accuracy(weight_decay_dict, model_name, epoch_num_list)
	best_weight = get_best_model_weight(weight_decay_dict)
	model = load(w2v, model_save_path.format(model_name, best_weight))[0]
	criterion = nn.BCEWithLogitsLoss()
	test_iterator = data_manager.get_torch_iterator(data_subset=TEST)
	test_acc, test_loss = evaluate(model, test_iterator, criterion)
	print("for model{}:\ntest accuracy is {}\ntest loss is {}".format(model_name, test_acc, test_loss))
	get_special_acc(model, dataset, test_iterator)
	return


def train_lstm_with_w2v(should_load=False):
	"""
	Here comes your code for training and evaluation of the LSTM model.
	"""
	dataset = data_loader.SentimentTreeBank()
	model_name = "lstm"
	pickle_path = "{}_pickled_weight_decay_dict.p".format(model_name)
	if not os.path.isdir("saved_models/{}".format(model_name)):
		os.mkdir("saved_models/{}".format(model_name))
	model_save_path = "saved_models/{0}/after_wd_{1}.pth"
	BATCH_SIZE = 64
	data_manager = DataManager(data_type=W2V_SEQUENCE, batch_size=BATCH_SIZE, embedding_dim=W2V_EMBEDDING_DIM)
	LEARNING_RATE = 1e-3
	weight_decay = 0.0001
	# weight_decay = 0
	DROPOUT = 0.5
	HIDDEN_LAYER_DIM = 100
	NUM_LAYERS = 1
	weight_decay_dict = {weight_decay: {}}
	N_EPOCHS = 4
	if not should_load:
		lstm = LSTM(embedding_dim=W2V_EMBEDDING_DIM, hidden_dim=HIDDEN_LAYER_DIM, n_layers=NUM_LAYERS, dropout=DROPOUT)
		val_accuracy_list, val_loss_list, train_accuracy_list, train_loss_list = train_model(lstm, data_manager,
		                                                                                     N_EPOCHS, LEARNING_RATE,
		                                                                                     weight_decay)
		save_model(lstm, model_save_path.format(model_name, weight_decay), 0)
		weight_decay_dict[weight_decay]["val_loss_list"] = val_loss_list
		weight_decay_dict[weight_decay]["val_accuracy_list"] = val_accuracy_list
		weight_decay_dict[weight_decay]["train_accuracy_list"] = train_accuracy_list
		weight_decay_dict[weight_decay]["train_loss_list"] = train_loss_list
		save_pickle(weight_decay_dict, pickle_path)
	else:
		weight_decay_dict = load_pickle(pickle_path)
		lstm = LSTM(embedding_dim=W2V_EMBEDDING_DIM, hidden_dim=HIDDEN_LAYER_DIM, n_layers=NUM_LAYERS, dropout=DROPOUT)
	epoch_num_list = np.arange(1, N_EPOCHS + 1)
	plot_weight_decay_loss_accuracy(weight_decay_dict, model_name, epoch_num_list)
	best_weight = get_best_model_weight(weight_decay_dict)
	model = load(lstm, model_save_path.format(model_name, best_weight))[0]
	criterion = nn.BCEWithLogitsLoss()
	test_iterator = data_manager.get_torch_iterator(data_subset=TEST)
	test_acc, test_loss = evaluate(model, test_iterator, criterion)
	print("for model{}:\ntest accuracy is {}\ntest loss is {}".format(model_name, test_acc, test_loss))
	get_special_acc(model, dataset, test_iterator)
	return


if __name__ == '__main__':
	# train_log_linear_with_one_hot()
	# train_log_linear_with_w2v(True)
	train_lstm_with_w2v(False)
