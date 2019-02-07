import tensorflow as tf
import tensorflow_hub as hub

class MultiHead_Model():
	def __init__(self, shared_layer_size = 1000, genus_count = 578,
					   species_count = 1890, quantize_layer = 0, is_training = 0,
					   learning_rate =  0.01):

		self.shared_layer_size = shared_layer_size
		self.genus_count = genus_count
		self.species_count = species_count
		self.learning_rate = learning_rate
		self.top_k = 1
		#loading pretrained Inception V3
		self._add_pretrained_inception()
		#adding our multihead model
		self._add_multi_head()
		#if we want to train add loss and optimizer
		if is_training: self._train()
		#adding evaluation subgraph
		self._evaluate()

	def _add_pretrained_inception(self, module_name = \
			'https://tfhub.dev/google/imagenet/inception_v3/feature_vector/1'):

		module_spec = hub.load_module_spec(module_name)
		self.model = hub.Module(module_spec)

		self.height, self.width = hub.get_expected_image_size(module_spec)

		resized_input_tensor = \
				tf.placeholder(tf.float32, [None, self.height, self.width, 3])
		bottleneck_tensor = self.model(resized_input_tensor)
		#this will be fed later since during training its 0.5, and during eval 1
		self.keep_prob = tf.placeholder(tf.float32)
		self.batch_size, self.bottleneck_tensor_size = \
										bottleneck_tensor.get_shape().as_list()
		assert self.batch_size is None, \
									'We want to work with arbitrary batch size.'

		with tf.name_scope('input'):
			self.bottleneck_input = tf.placeholder_with_default(
									bottleneck_tensor,
									shape=[self.batch_size, self.bottleneck_tensor_size],
									name='BottleneckInputPlaceholder')
			self.dropout_layer = tf.nn.dropout(self.bottleneck_input, self.keep_prob)


	def _add_single_layer(self):
		'''
		Adding a feed-forward network to Inception
		'''

		self.ground_truth_species_input = tf.placeholder(
		   					tf.int64, [batch_size], name='GroundTruthInput')

		# Organizing the following ops so they are easier to see in TensorBoard.
		with tf.name_scope('final_retrain_ops'):
			with tf.name_scope('weights'):
		 		initial_value = tf.truncated_normal(
		     					[self.bottleneck_tensor_size,
								 self.shared_layer_size], stddev=0.001)
		 		self.layer_weights = tf.Variable(initial_value, name='inter_weights')
		 		initial_value2 = tf.truncated_normal([self.shared_layer_size, self.species_count])
		 		self.hidden_layer_w = tf.Variable(initial_value2, name='final_weights')
		 		self.variable_summaries(self.layer_weights)
				self.variable_summaries(self.hidden_layer_w)

		with tf.name_scope('biases'):
			self.hidden_biases = tf.Variable(tf.zeros([self.shared_layer_size]), name='hidden_biases')
		 	self.layer_biases = tf.Variable(tf.zeros([self.species_count]), name='final_biases')
		 	variable_summaries(self.hidden_biases, self.layer_biases)

		with tf.name_scope('Wx_plus_b'):
			self.hidden_logits = tf.matmul(self.dropout_layer, self.layer_weights) + self.hidden_biases
			self.dropout_layer2 = tf.nn.dropout(self.hidden_logits, self.keep_prob)
			self.logits = tf.matmul(self.dropout_layer2, self.hidden_layer_w) + self.layer_biases
			tf.summary.histogram('pre_activations', self.logits)

		self.final_tensor = tf.nn.softmax(self.logits, name='final_species_tensor')

		# The tf.contrib.quantize functions rewrite the graph in place for
		# quantization. The imported model graph has already been rewritten, so upon
		# calling these rewrites, only the newly added final layer will be
		# transformed.
		if quantize_layer:
			if is_training:
		 		tf.contrib.quantize.create_training_graph()
		else:
			tf.contrib.quantize.create_eval_graph()

		tf.summary.histogram('activations', self.final_tensor)


		with tf.name_scope('cross_entropy'):
			cross_entropy_mean = tf.losses.sparse_softmax_cross_entropy(
		   			labels=self.ground_truth_species_input, logits=self.logits)

		tf.summary.scalar('cross_entropy', cross_entropy_mean)

		with tf.name_scope('train'):
			self.optimizer = tf.train.GradientDescentOptimizer(self.learning_rate)
			self.train_step = optimizer.minimize(cross_entropy_mean)



	def _add_multi_head(self):
		'''
		This will add dropout from the image embedding to the first shared layer
		This shared layer will in turn go to two separate layers, each predicting
		either the genus or species of the picture
		'''
		# Organizing the following ops so they are easier to see in TensorBoard.
		with tf.name_scope('final_retrain_ops'):
			with tf.name_scope('weights'):
				initial_shared_value = tf.truncated_normal(
						[self.bottleneck_tensor_size, self.shared_layer_size], stddev=0.001)
				initial_genus_value = tf.truncated_normal(
						[self.shared_layer_size, self.genus_count], stddev=0.001)
				intial_species_value = tf.truncated_normal(
						[self.shared_layer_size, self.species_count], stddev=0.001)

				self.shared_layer_weights = tf.Variable(initial_shared_value,
													name='shared_layer_weights')
				self.genus_layer_weights = tf.Variable(initial_genus_value,
													name='final_genus_weights')
				self.species_layer_weights = tf.Variable(intial_species_value,
													name='final_species_weights')
				self.variable_summaries(self.shared_layer_weights)
				self.variable_summaries(self.genus_layer_weights)
				self.variable_summaries(self.species_layer_weights)

		with tf.name_scope('biases'):
			self.shared_layer_biases = tf.Variable(tf.zeros([self.shared_layer_size]),
													name='shared_layer_biases')
			self.layer_genus_biases = tf.Variable(tf.zeros([self.genus_count]),
													name='final_genus_biases')
			self.layer_species_biases = tf.Variable(tf.zeros([self.species_count]),
													name='final_species_biases')
			self.variable_summaries(self.shared_layer_biases)
			self.variable_summaries(self.layer_genus_biases)
			self.variable_summaries(self.layer_species_biases)

		with tf.name_scope('Wx_plus_b'):
			self.shared_logits = \
					tf.matmul(self.dropout_layer, self.shared_layer_weights) \
													+ self.shared_layer_biases
			self.genus_logits = \
					tf.matmul(self.shared_logits, self.genus_layer_weights) \
													+ self.layer_genus_biases
			self.species_logits = \
					tf.matmul(self.shared_logits, self.species_layer_weights) \
													+ self.layer_species_biases
			#attach summaries for tensorboard
			tf.summary.histogram('pre_activations_shared', self.shared_logits)
			tf.summary.histogram('pre_activations_genus', self.genus_logits)
			tf.summary.histogram('pre_activations_species', self.species_logits)

		self.final_genus_tensor = \
				tf.nn.softmax(self.genus_logits, name='final_genus_tensor')
		self.final_species_tensor = \
				tf.nn.softmax(self.species_logits, name='final_species_tensor')

		tf.summary.histogram('activations_genus', self.final_genus_tensor)
		tf.summary.histogram('activations_species', self.final_species_tensor)

		self.ground_truth_genus_input = tf.placeholder(
			tf.int64, [self.batch_size], name='GroundTruthGenusInput')
		self.ground_truth_species_input = tf.placeholder(
			tf.int64, [self.batch_size], name='GroundTruthSpeciesInput')
		with tf.name_scope('cross_entropy'):
			self.cross_entropy_mean_genus = \
										tf.losses.sparse_softmax_cross_entropy(
										labels=self.ground_truth_genus_input,
										logits=self.genus_logits)
			self.cross_entropy_mean_species = \
										tf.losses.sparse_softmax_cross_entropy(
										labels=self.ground_truth_species_input,
										logits=self.species_logits)
			self.LOSS = self.cross_entropy_mean_genus \
							   + self.cross_entropy_mean_species

		tf.summary.scalar('cross_entropy_genus', self.cross_entropy_mean_genus)
		tf.summary.scalar('cross_entropy_species', self.cross_entropy_mean_species)
		tf.summary.scalar('cross_entropy', self.LOSS)

	def _train(self):
		with tf.name_scope('train'):
			self.optimizer = \
						tf.train.GradientDescentOptimizer(self.learning_rate)
			self.train_step = self.optimizer.minimize(self.LOSS)

	def _evaluate(self):
		with tf.name_scope('accuracy'):
			with tf.name_scope('correct_prediction'):
				if self.top_k == 1:
					self.genus_prediction = \
										tf.argmax(self.final_genus_tensor, 1)
					self.species_prediction = \
										tf.argmax(self.final_species_tensor, 1)
				elif self.top_k > 1:
					self.predictions = \
						tf.nn.top_k(result_tensor, k=self.top_k, sorted=True, name=None)
			self.genus_correct_prediction = \
						tf.equal(self.genus_prediction,
								 self.ground_truth_genus_input)
			self.species_correct_prediction = \
						tf.equal(self.species_prediction,
								 self.ground_truth_species_input)
			with tf.name_scope('accuracy'):
				self.genus_evaluation_step = \
					tf.reduce_mean(tf.cast(self.genus_correct_prediction, tf.float32))
				self.species_evaluation_step = \
					tf.reduce_mean(tf.cast(self.species_correct_prediction, tf.float32))
			tf.summary.scalar('Genus_accuracy', self.genus_evaluation_step)
			tf.summary.scalar('Species_accuracy', self.species_evaluation_step)


	def _add_LSTM(self):
		resized_input_tensor = tf.placeholder(tf.float32, [None, height, width, 3])
		m = hub.Module(module_spec)
		bottleneck_tensor = m(resized_input_tensor)
		batch_size, bottleneck_tensor_size = bottleneck_tensor.get_shape().as_list()

		#word embedding inputs
		word_embedding = tf.Variable(tf.random_uniform([n_words, dim_embed], -0.1, 0.1), name='word_embedding')
		embedding_bias = tf.Variable(tf.zeros([dim_embed]), name='embedding_bias')

		#define LSTM
		lstm = tf.contrib.rnn.LSTMCell(dim_hidden)

		#image embedding inputs
		img_embedding = tf.Variable(tf.random_uniform([bottleneck_tensor_size, dim_hidden], -0.1, 0.1), name='img_embedding')
		img_embedding_bias = tf.Variable(tf.zeros([dim_hidden]), name='img_embedding_bias')

		#LSTM output to word embedding output
		word_encoding = tf.Variable(tf.random_uniform([dim_hidden, n_words], -0.1, 0.1), name='word_encoding')
		word_encoding_bias = tf.Variable(tf.zeros([n_words]), name='word_encoding_bias')

		# (describes how long our caption is with an array of 0/1 values of length `maxlen`
		img = tf.placeholder(tf.float32, [batch_size, bottleneck_tensor_size])
		caption_placeholder = tf.placeholder(tf.int32, [batch_size, n_lstm_steps])
		mask = tf.placeholder(tf.float32, [batch_size, n_lstm_steps])

		# getting an initial LSTM embedding from our image_imbedding
		image_embedding = tf.matmul(img, img_embedding) + img_embedding_bias

		# setting initial state of our LSTM
		state = lstm.zero_state(batch_size, dtype=tf.float32)

		total_loss = 0.0
		with tf.variable_scope("RNN"):
			for i in range(n_lstm_steps):
				if i > 0:
					'''if this isnt the first iteration of our LSTM we need to get
					the word_embedding corresponding to the (i-1)th word in our caption'''
					with tf.device("/cpu:0"):
						current_embedding = \
							tf.nn.embedding_lookup(word_embedding,
													caption_placeholder[:,i-1]) \
																+ embedding_bias
				else:
					 #if this is the first iteration of our LSTM we utilize the embedded image as our input
					current_embedding = image_embedding
				if i > 0:
					# allows us to reuse the LSTM tensor variable on each iteration
					tf.get_variable_scope().reuse_variables()

				out, state = lstm(current_embedding, state)


				if i > 0:
					#get the one-hot representation of the next word in our caption
					labels = tf.expand_dims(caption_placeholder[:, i], 1)
					ix_range=tf.range(0, batch_size, 1)
					ixs = tf.expand_dims(ix_range, 1)
					concat = tf.concat([ixs, labels],1)
					onehot = tf.sparse_to_dense(
							concat, tf.stack([batch_size, n_words]), 1.0, 0.0)


					#perform a softmax classification to generate the next word in the caption
					logit = tf.matmul(out, word_encoding) + word_encoding_bias
					cross_entropy = tf.nn.softmax_cross_entropy_with_logits(logits=logit, labels=onehot)
					cross_entropy = cross_entropy * mask[:,i]

					loss = tf.reduce_sum(cross_entropy)
					total_loss += loss

			total_loss = total_loss / tf.reduce_sum(mask[:,1:])

	def variable_summaries(self, var):
		"""Attach a lot of summaries to a Tensor (for TensorBoard visualization)."""
		with tf.name_scope('summaries'):
			mean = tf.reduce_mean(var)
			tf.summary.scalar('mean', mean)
		with tf.name_scope('stddev'):
			stddev = tf.sqrt(tf.reduce_mean(tf.square(var - mean)))
			tf.summary.scalar('stddev', stddev)
		tf.summary.scalar('max', tf.reduce_max(var))
		tf.summary.scalar('min', tf.reduce_min(var))
		tf.summary.histogram('histogram', var)
