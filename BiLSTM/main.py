import numpy as np
from python_speech_features import mfcc
import scipy.io.wavfile as wav
import os
import time
import tensorflow as tf
from tensorflow.python.ops import ctc_ops
from collections import Counter


def get_wav_files(wav_path):
    '''
    Fetch all the wav files in a folder
    :param wav_path: path to the folder that contains all the wav files
    :return: list of wav files
    '''
    wav_files = []
    for (dirpath, dirnames, filenames) in os.walk(wav_path):
        for filename in filenames:
            if filename.endswith('.wav') or filename.endswith('.WAV'):
                # print(filename)
                filename_path = os.path.join(dirpath, filename)
                # print(filename_path)
                wav_files.append(filename_path)
    return wav_files


def get_tran_texts(wavfiles, tran_path):
    '''
    Fetch wav transcripts
    :param wavfiles: list a wav files
    :param tran_path: path to the folder that contains all the trn files
    :return:
    '''
    tran_texts = []
    wav_files = []
    for wav_file in wavfiles:
        (wav_path, wav_filename) = os.path.split(wav_file)
        tran_file = os.path.join(tran_path, wav_filename + '.trn')
        # print(tran_file)
        if os.path.exists(tran_file) is False:
            return None

        fd = open(tran_file, 'r', encoding='UTF-8')
        text = fd.readline()
        wav_files.append(wav_file)
        # 不知为何原数据中的文字中有很多空格，去除空格干扰因子（通俗的解释就是空格的MFCC特性种类太多，因为任何两个文字语音间的空格特征都不一样，模型无法定位空格到底长啥样，导致模型收敛极慢甚至不收敛）
        tran_texts.append(text.split('\n')[0].replace(' ', ''))
        fd.close()
    return wav_files, tran_texts


# 获取wav和对应的翻译文字
def get_wav_files_and_tran_texts(wav_path, tran_path):
    """
    Get wav files amd transcripts
    :param wav_path: folder that contains wav files
    :param tran_path: folder that contains trn files
    :return:
    """
    wavfiles = get_wav_files(wav_path)
    wav_files, tran_texts = get_tran_texts(wavfiles, tran_path)

    return wav_files, tran_texts


def sparse_tuple_to_texts_ch(tuple, words):
    """
    Convert typle to text
    :param tuple:
    :param words:
    :return:
    """
    indices = tuple[0]
    values = tuple[1]
    results = [''] * tuple[2][0]
    for i in range(len(indices)):
        index = indices[i][0]
        c = words[values[i]]
        results[index] = results[index] + c
    return results


# 将密集矩阵的字向量转成文字
def ndarray_to_text_ch(value, words):
    results = ''
    for i in range(len(value)):
        if value[i] == len(words):
            results += ' '
        else:
            results += words[value[i]]
    return results


# 创建序列的稀疏表示
def sparse_tuple_from(sequences, dtype=np.int32):
    indices = []
    values = []
    for n, seq in enumerate(sequences):
        indices.extend(zip([n] * len(seq), range(len(seq))))
        values.extend(seq)
    indices = np.asarray(indices, dtype=np.int64)
    values = np.asarray(values, dtype=dtype)
    shape = np.asarray([len(sequences), indices.max(0)[1] + 1], dtype=np.int64)
    # return tf.SparseTensor(indices=indices, values=values, shape=shape)
    return indices, values, shape


# 将音频数据转为时间序列（列）和MFCC（行）的矩阵，将对应的译文转成字向量
def get_audio_and_transcriptch(txt_files, wav_files, n_input, n_context, word_num_map, txt_labels=None):
    audio = []
    audio_len = []
    transcript = []
    transcript_len = []
    if txt_files != None:
        txt_labels = txt_files
    for txt_obj, wav_file in zip(txt_labels, wav_files):
        # load audio and convert to features
        audio_data = audiofile_to_input_vector(wav_file, n_input, n_context)
        audio_data = audio_data.astype('float32')
        # print(word_num_map)
        audio.append(audio_data)
        audio_len.append(np.int32(len(audio_data)))
        # load text transcription and convert to numerical array
        target = []
        if txt_files != None:  # txt_obj是文件
            target = get_ch_lable_v(txt_obj, word_num_map)
        else:
            target = get_ch_lable_v(None, word_num_map, txt_obj)  # txt_obj是labels
        # target = text_to_char_array(target)
        transcript.append(target)
        transcript_len.append(len(target))
    audio = np.asarray(audio)
    audio_len = np.asarray(audio_len)
    transcript = np.asarray(transcript)
    transcript_len = np.asarray(transcript_len)
    return audio, audio_len, transcript, transcript_len


# 将字符转成向量，其实就是根据字找到字在word_num_map中所应对的下标
def get_ch_lable_v(txt_file, word_num_map, txt_label=None):
    words_size = len(word_num_map)
    to_num = lambda word: word_num_map.get(word, words_size)
    if txt_file != None:
        txt_label = get_ch_lable(txt_file)
    # print(txt_label)
    labels_vector = list(map(to_num, txt_label))
    # print(labels_vector)
    return labels_vector


def get_ch_lable(txt_file):
    labels = ""
    with open(txt_file, 'rb') as f:
        for label in f:
            labels = labels + label.decode('gb2312')
    return labels


# 将音频信息转成MFCC特征
# 参数说明---audio_filename：音频文件   numcep：梅尔倒谱系数个数
#       numcontext：对于每个时间段，要包含的上下文样本个数
def audiofile_to_input_vector(audio_filename, numcep, numcontext):
    """
    Convert speech waveform to MFCC features
    :param audio_filename:
    :param numcep: numeber of Cepstrum
    :param numcontext: number of context
    :return:
    """
    # load audio file
    fs, audio = wav.read(audio_filename)
    # Get MFCC Coefficient
    orig_inputs = mfcc(audio, samplerate=fs, numcep=numcep)

    orig_inputs = orig_inputs[::2]  # (478, 26)
    train_inputs = np.array([], np.float32)
    train_inputs.resize((orig_inputs.shape[0], numcep + 2 * numcep * numcontext))

    # Prepare pre-fix post fix context
    empty_mfcc = np.array([])
    empty_mfcc.resize((numcep))

    # Prepare train_inputs with past and future contexts
    time_slices = range(train_inputs.shape[0])

    # context_past_min和context_future_ma to check for samples that requires padding
    context_past_min = time_slices[0] + numcontext
    context_future_max = time_slices[-1] - numcontext
    for time_slice in time_slices:
        need_empty_past = max(0, (context_past_min - time_slice))
        empty_source_past = list(empty_mfcc for empty_slots in range(need_empty_past))
        data_source_past = orig_inputs[max(0, time_slice - numcontext):time_slice]
        assert (len(empty_source_past) + len(data_source_past) == numcontext)
        need_empty_future = max(0, (time_slice - context_future_max))
        empty_source_future = list(empty_mfcc for empty_slots in range(need_empty_future))
        data_source_future = orig_inputs[time_slice + 1:time_slice + numcontext + 1]
        assert (len(empty_source_future) + len(data_source_future) == numcontext)

        if need_empty_past:
            past = np.concatenate((empty_source_past, data_source_past))
        else:
            past = data_source_past

        if need_empty_future:
            future = np.concatenate((data_source_future, empty_source_future))
        else:
            future = data_source_future

        past = np.reshape(past, numcontext * numcep)
        now = orig_inputs[time_slice]
        future = np.reshape(future, numcontext * numcep)
        train_inputs[time_slice] = np.concatenate((past, now, future))
        assert (len(train_inputs[time_slice]) == numcep + 2 * numcep * numcontext)

    train_inputs = (train_inputs - np.mean(train_inputs)) / np.std(train_inputs)
    return train_inputs

# Pad Sequence to same length
def pad_sequences(sequences, maxlen=None, dtype=np.float32,
                  padding='post', truncating='post', value=0.):
    # [478 512 503 406 481 509 422 465]
    lengths = np.asarray([len(s) for s in sequences], dtype=np.int64)
    nb_samples = len(sequences)

    if maxlen is None:
        maxlen = np.max(lengths)

    sample_shape = tuple()
    for s in sequences:
        if len(s) > 0:
            sample_shape = np.asarray(s).shape[1:]
            break

    x = (np.ones((nb_samples, maxlen) + sample_shape) * value).astype(dtype)
    for idx, s in enumerate(sequences):
        if len(s) == 0:
            continue  # 序列为空，跳过

        # post means post zero-padding
        # pre means pre zero-padding
        if truncating == 'pre':
            trunc = s[-maxlen:]
        elif truncating == 'post':
            trunc = s[:maxlen]
        else:
            raise ValueError('Truncating type "%s" not understood' % truncating)

        # check `trunc` has expected shape
        trunc = np.asarray(trunc, dtype=dtype)
        if trunc.shape[1:] != sample_shape:
            raise ValueError('Shape of sample %s of sequence at position %s is different from expected shape %s' %
                             (trunc.shape[1:], idx, sample_shape))

        if padding == 'post':
            x[idx, :len(trunc)] = trunc
        elif padding == 'pre':
            x[idx, -len(trunc):] = trunc
        else:
            raise ValueError('Padding type "%s" not understood' % padding)

    return x, lengths



wav_path = 'D:\\Individual_Project\\data_thchs30\\train'
label_file = 'D:\\Individual_Project\\data_thchs30\\data'
# wav_files, labels = get_wavs_lables(wav_path,label_file)
wav_files, labels = get_wav_files_and_tran_texts(wav_path, label_file)

wav_files = wav_files[:2000]
labels = labels[:2000]

# Pronunciation Dictionary
all_words = []
for label in labels:

    all_words += [word for word in label]
counter = Counter(all_words)
words = sorted(counter)
words += [""]
words_size = len(words)
word_num_map = dict(zip(words, range(words_size)))
print('字表大小:', words_size)

n_input = 26
n_context = 9
batch_size = 8


def next_batch(wav_files, labels, start_idx=0, batch_size=1):
    filesize = len(labels)

    end_idx = min(filesize, start_idx + batch_size)
    idx_list = range(start_idx, end_idx)

    txt_labels = [labels[i] for i in idx_list]
    wav_files = [wav_files[i] for i in idx_list]
    # 将音频文件转成要训练的数据
    (source, audio_len, target, transcript_len) = get_audio_and_transcriptch(None,
                                                                             wav_files,
                                                                             n_input,
                                                                             n_context, word_num_map, txt_labels)

    start_idx += batch_size
    # Verify that the start_idx is not largVerify that the start_idx is not ler than total available sample size
    if start_idx >= filesize:
        start_idx = -1

    # Pad input to max_time_step of this batch
    source, source_lengths = pad_sequences(source)
    sparse_labels = sparse_tuple_from(target)

    return start_idx, source, source_lengths, sparse_labels


print('Wav File:  ' + wav_files[0])
print('Transcript:  ' + labels[0])
# Fetch one batch
next_idx, source, source_len, sparse_lab = next_batch(wav_files, labels, 0, batch_size)
print(np.shape(source))

t = sparse_tuple_to_texts_ch(sparse_lab, words)
print(t[0])



b_stddev = 0.046875
h_stddev = 0.046875

n_hidden = 1024
n_hidden_1 = 1024
n_hidden_2 = 1024
n_hidden_5 = 1024
n_cell_dim = 1024
n_hidden_3 = 2 * 1024

keep_dropout_rate = 0.95
relu_clip = 20

"""
used to create a variable in CPU memory.
"""


def variable_on_gpu(name, shape, initializer):
    # Use the /gpu:0 device for scoped operations
    with tf.device('/gpu:0'):
        # Create or get apropos variable
        var = tf.get_variable(name=name, shape=shape, initializer=initializer)
    return var


def BiRNN_model(batch_x, seq_length, n_input, n_context, n_character, keep_dropout):
    # batch_x_shape: [batch_size, amax_stepsize, n_input + 2 * n_input * n_context]
    batch_x_shape = tf.shape(batch_x)


    batch_x = tf.transpose(batch_x, [1, 0, 2])
    batch_x = tf.reshape(batch_x, [-1, n_input + 2 * n_input * n_context])

    # Clipped RELU activation and dropout.
    # 1st layer
    with tf.name_scope('fc1'):
        b1 = variable_on_gpu('b1', [n_hidden_1], tf.random_normal_initializer(stddev=b_stddev))
        h1 = variable_on_gpu('h1', [n_input + 2 * n_input * n_context, n_hidden_1],
                             tf.random_normal_initializer(stddev=h_stddev))
        layer_1 = tf.minimum(tf.nn.relu(tf.add(tf.matmul(batch_x, h1), b1)), relu_clip)
        layer_1 = tf.nn.dropout(layer_1, keep_dropout)

    # 2nd layer
    with tf.name_scope('fc2'):
        b2 = variable_on_gpu('b2', [n_hidden_2], tf.random_normal_initializer(stddev=b_stddev))
        h2 = variable_on_gpu('h2', [n_hidden_1, n_hidden_2], tf.random_normal_initializer(stddev=h_stddev))
        layer_2 = tf.minimum(tf.nn.relu(tf.add(tf.matmul(layer_1, h2), b2)), relu_clip)
        layer_2 = tf.nn.dropout(layer_2, keep_dropout)

    # 3rd layer
    with tf.name_scope('fc3'):
        b3 = variable_on_gpu('b3', [n_hidden_3], tf.random_normal_initializer(stddev=b_stddev))
        h3 = variable_on_gpu('h3', [n_hidden_2, n_hidden_3], tf.random_normal_initializer(stddev=h_stddev))
        layer_3 = tf.minimum(tf.nn.relu(tf.add(tf.matmul(layer_2, h3), b3)), relu_clip)
        layer_3 = tf.nn.dropout(layer_3, keep_dropout)

    # 双向rnn
    with tf.name_scope('lstm'):
        # Forward direction cell:
        lstm_fw_cell = tf.contrib.rnn.BasicLSTMCell(n_cell_dim, forget_bias=1.0, state_is_tuple=True)
        lstm_fw_cell = tf.contrib.rnn.DropoutWrapper(lstm_fw_cell,
                                                     input_keep_prob=keep_dropout)
        # Backward direction cell:
        lstm_bw_cell = tf.contrib.rnn.BasicLSTMCell(n_cell_dim, forget_bias=1.0, state_is_tuple=True)
        lstm_bw_cell = tf.contrib.rnn.DropoutWrapper(lstm_bw_cell,
                                                     input_keep_prob=keep_dropout)

        # `layer_3`  `[amax_stepsize, batch_size, 2 * n_cell_dim]`
        layer_3 = tf.reshape(layer_3, [-1, batch_x_shape[0], n_hidden_3])

        outputs, output_states = tf.nn.bidirectional_dynamic_rnn(cell_fw=lstm_fw_cell,
                                                                 cell_bw=lstm_bw_cell,
                                                                 inputs=layer_3,
                                                                 dtype=tf.float32,
                                                                 time_major=True,
                                                                 sequence_length=seq_length)

        # Connect forward backward result [amax_stepsize, batch_size, 2 * n_cell_dim]
        outputs = tf.concat(outputs, 2)
        # to a single tensor of shape [amax_stepsize * batch_size, 2 * n_cell_dim]
        outputs = tf.reshape(outputs, [-1, 2 * n_cell_dim])

    with tf.name_scope('fc5'):
        b5 = variable_on_gpu('b5', [n_hidden_5], tf.random_normal_initializer(stddev=b_stddev))
        h5 = variable_on_gpu('h5', [(2 * n_cell_dim), n_hidden_5], tf.random_normal_initializer(stddev=h_stddev))
        layer_5 = tf.minimum(tf.nn.relu(tf.add(tf.matmul(outputs, h5), b5)), relu_clip)
        layer_5 = tf.nn.dropout(layer_5, keep_dropout)

    with tf.name_scope('fc6'):
        # 全连接层用于softmax分类
        b6 = variable_on_gpu('b6', [n_character], tf.random_normal_initializer(stddev=b_stddev))
        h6 = variable_on_gpu('h6', [n_hidden_5, n_character], tf.random_normal_initializer(stddev=h_stddev))
        layer_6 = tf.add(tf.matmul(layer_5, h6), b6)

    # 将2维[amax_stepsize * batch_size, n_character]转成3维 time-major [amax_stepsize, batch_size, n_character].
    layer_6 = tf.reshape(layer_6, [-1, batch_x_shape[0], n_character])
    print('n_character:' + str(n_character))
    # Output shape: [amax_stepsize, batch_size, n_character]
    return layer_6



input_tensor = tf.placeholder(tf.float32, [None, None, n_input + (2 * n_input * n_context)], name='input')
# Use sparse_placeholder; will generate a SparseTensor, required by ctc_loss op.

targets = tf.sparse_placeholder(tf.int32, name='targets')
seq_length = tf.placeholder(tf.int32, [None], name='seq_length')
keep_dropout = tf.placeholder(tf.float32)

# logits is the non-normalized output/activations from the last layer.
# logits will be input for the loss function.
# nn_model is from the import statement in the load_model function
logits = BiRNN_model(input_tensor, tf.to_int64(seq_length), n_input, n_context, words_size + 1, keep_dropout)

# Loss function: CTC Loss
avg_loss = tf.reduce_mean(ctc_ops.ctc_loss(targets, logits, seq_length))

# Optimiser
learning_rate = 0.001
optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(avg_loss)

# CTC Decoder - standard beam search algorithm
with tf.name_scope("decode"):
    decoded, log_prob = ctc_ops.ctc_beam_search_decoder(logits, seq_length, merge_repeated=False)


with tf.name_scope("accuracy"):
    distance = tf.edit_distance(tf.cast(decoded[0], tf.int32), targets)
    # 计算label error rate (accuracy)
    ler = tf.reduce_mean(distance, name='label_error_rate')


epochs = 100
# 模型保存地址
savedir = "saver/"

if os.path.exists(savedir) == False:
    os.mkdir(savedir)

saver = tf.train.Saver(max_to_keep=1)

with tf.Session() as sess:
    # Initialisation
    sess.run(tf.global_variables_initializer())
    kpt = tf.train.latest_checkpoint(savedir)
    print("kpt:", kpt)
    startepo = 0
    if kpt != None:
        saver.restore(sess, kpt)
        ind = kpt.find("-")
        startepo = int(kpt[ind + 1:])
        print(startepo)

    section = '\n{0:=^40}\n'
    print(section.format('Run training epoch'))

    train_start = time.time()
    floss = open("loss_value", 'w', encoding='UTF-8')
    for epoch in range(epochs):
        epoch_start = time.time()
        if epoch < startepo:
            continue
        print("epoch start:", epoch, "total epochs= ", epochs)
        #######################run batch####
        n_batches_per_epoch = int(np.ceil(len(labels) / batch_size))
        print("total loop ", n_batches_per_epoch, "in one epoch，", batch_size, "items in one loop")
        train_cost = 0
        train_ler = 0
        next_idx = 0
        for batch in range(n_batches_per_epoch):

            print('Start loading:' + str(batch))
            next_idx, source, source_lengths, sparse_labels = next_batch(wav_files, labels, next_idx, batch_size)
            print('Finish loading')
            feed = {input_tensor: source, targets: sparse_labels, seq_length: source_lengths,
                    keep_dropout: keep_dropout_rate}
            # Calculate avg_loss optimizer ;
            batch_cost, _ = sess.run([avg_loss, optimizer], feed_dict=feed)
            train_cost += batch_cost

            if (batch + 1) % n_batches_per_epoch == 0:
                print('loop:', batch, 'Train cost: ', train_cost / (batch + 1))
                print('loop:', batch, 'Train cost: ', train_cost / (batch + 1), file=floss)
                feed2 = {input_tensor: source, targets: sparse_labels, seq_length: source_lengths, keep_dropout: 1.0}
                d, train_ler = sess.run([decoded[0], ler], feed_dict=feed2)
                dense_decoded = tf.sparse_tensor_to_dense(d, default_value=-1).eval(session=sess)
                dense_labels = sparse_tuple_to_texts_ch(sparse_labels, words)
                counter = 0
                print('Label err rate: ', train_ler)
                for orig, decoded_arr in zip(dense_labels, dense_decoded):
                    # convert to strings
                    decoded_str = ndarray_to_text_ch(decoded_arr, words)
                    print(' file {}'.format(counter))
                    print('Original: {}'.format(orig))
                    print('Decoded:  {}'.format(decoded_str))
                    counter = counter + 1
                    break

            if (batch + 1) % 100 == 0:
                saver.save(sess, savedir + "saver.cpkt", global_step=epoch)
        epoch_duration = time.time() - epoch_start
        log = 'Epoch {}/{}, train_cost: {:.3f}, train_ler: {:.3f}, time: {:.2f} sec'
        print(log.format(epoch, epochs, train_cost, train_ler, epoch_duration))
    train_duration = time.time() - train_start
    print('Training complete, total duration: {:.2f} min'.format(train_duration / 60))