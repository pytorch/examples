import functools
import os
import re
from functools import reduce
from itertools import chain

import numpy as np
import torch
from sklearn import model_selection
from sklearn.metrics import jaccard_similarity_score
from torch.autograd import Variable

long_tensor_type = torch.LongTensor
float_tensor_type = torch.FloatTensor

if (torch.cuda.is_available()):
    long_tensor_type = torch.cuda.LongTensor
    float_tensor_type = torch.cuda.FloatTensor


def process_data(args):
    test_size = .1
    random_state = None
    data, test_data, vocab = load_data(args.data_dir, args.joint_training, args.task_number)

    '''
    data is of the form:
    [
        (
            [
                ['mary', 'moved', 'to', 'the', 'bathroom'], 
                ['john', 'went', 'to', 'the', 'hallway']
            ], 
            ['where', 'is', 'mary'], 
            ['bathroom']
        ),
        .
        .
        .
        ()
    ]
    '''

    memory_size, sentence_size, vocab_size, word_idx = calculate_parameter_values(data=data, debug=args.debug,
                                                                                  memory_size=args.memory_size,
                                                                                  vocab=vocab)

    train_set, train_batches, val_set, val_batches, test_set, test_batches = \
        vectorize_task_data(args.batch_size, data, args.debug, memory_size,
                            random_state, sentence_size, test_data, test_size, word_idx)

    return train_batches, val_batches, test_batches, train_set, val_set, test_set, \
           sentence_size, vocab_size, memory_size, word_idx


def load_data(data_dir, joint_training, task_number):
    if (joint_training == 0):
        start_task = task_number
        end_task = task_number
    else:
        start_task = 1
        end_task = 20

    train_data = []
    test_data = []

    while start_task <= end_task:
        task_train, task_test = load_task(data_dir, start_task)
        train_data += task_train
        test_data += task_test
        start_task += 1

    data = train_data + test_data

    vocab = sorted(reduce(lambda x, y: x | y, (set(list(chain.from_iterable(s)) + q + a) for s, q, a in data)))

    return train_data, test_data, vocab


def load_task(data_dir, task_id, only_supporting=False):
    '''
    Load the nth task. There are 20 tasks in total.

    Returns a tuple containing the training and testing data for the task.
    '''
    assert task_id > 0 and task_id < 21

    files = os.listdir(data_dir)
    files = [os.path.join(data_dir, f) for f in files]
    s = 'qa{}_'.format(task_id)
    train_file = [f for f in files if s in f and 'train' in f][0]
    test_file = [f for f in files if s in f and 'test' in f][0]
    train_data = get_stories(train_file, only_supporting)
    test_data = get_stories(test_file, only_supporting)
    return train_data, test_data


def get_stories(f, only_supporting=False):
    '''
    Given a file name, read the file, retrieve the stories, and then convert the sentences into a single story.
    If max_length is supplied, any stories longer than max_length tokens will be discarded.
    '''
    with open(f) as f:
        return parse_stories(f.readlines(), only_supporting=only_supporting)


def parse_stories(lines, only_supporting=False):
    '''
    Parse stories provided in the bAbI tasks format
    If only_supporting is true, only the sentences that support the answer are kept.
    '''
    data = []
    story = []
    for line in lines:
        line = str.lower(line)
        nid, line = line.split(' ', 1)
        nid = int(nid)
        if nid == 1:
            story = []
        if '\t' in line:  # question
            q, a, supporting = line.split('\t')
            q = tokenize(q)
            # a = tokenize(a)
            # answer is one vocab word even if it's actually multiple words
            a = [a]
            substory = None

            # remove question marks
            if q[-1] == "?":
                q = q[:-1]

            if only_supporting:
                # Only select the related substory
                supporting = map(int, supporting.split(''))
                substory = [story[i - 1] for i in supporting]
            else:
                # Provide all the substories
                substory = [x for x in story if x]

            data.append((substory, q, a))
            story.append('')
        else:  # regular sentence
            # remove periods
            sent = tokenize(line)
            if sent[-1] == ".":
                sent = sent[:-1]
            story.append(sent)
    return data


def tokenize(sent):
    '''
    Return the tokens of a sentence including punctuation.
    >>> tokenize('Bob dropped the apple. Where is the apple?')
    ['Bob', 'dropped', 'the', 'apple', '.', 'Where', 'is', 'the', 'apple', '?']
    '''
    return [x.strip() for x in re.split('(\W+)?', sent) if x.strip()]


def calculate_parameter_values(data, debug, memory_size, vocab):
    word_idx = dict((c, i + 1) for i, c in enumerate(vocab))
    max_story_size = max(map(len, (s for s, _, _ in data)))
    mean_story_size = int(np.mean(list(map(len, (s for s, _, _ in data)))))
    sentence_size = max(map(len, chain.from_iterable(s for s, _, _ in data)))
    query_size = max(map(len, (q for _, q, _ in data)))
    memory_size = min(memory_size, max_story_size)
    vocab_size = len(word_idx) + 1  # +1 for nil word
    sentence_size = max(query_size, sentence_size)  # for the position
    if debug is True:
        print("Longest sentence length: ", sentence_size)
        print("Longest story length: ", max_story_size)
        print("Average story length: ", mean_story_size)
        print("Average memory size: ", memory_size)
    return memory_size, sentence_size, vocab_size, word_idx


def vectorize_task_data(batch_size, data, debug, memory_size, random_state, sentence_size, test,
                        test_size, word_idx):
    S, Q, Y = vectorize_data(data, word_idx, sentence_size, memory_size)

    if debug is True:
        print("S : ", S)
        print("Q : ", Q)
        print("Y : ", Y)
    trainS, valS, trainQ, valQ, trainY, valY = model_selection.train_test_split(S, Q, Y, test_size=test_size,
                                                                                random_state=random_state)
    testS, testQ, testY = vectorize_data(test, word_idx, sentence_size, memory_size)

    if debug is True:
        print(S[0].shape, Q[0].shape, Y[0].shape)
        print("Training set shape", trainS.shape)

    # params
    n_train = trainS.shape[0]
    n_val = valS.shape[0]
    n_test = testS.shape[0]
    if debug is True:
        print("Training Size: ", n_train)
        print("Validation Size: ", n_val)
        print("Testing Size: ", n_test)
    train_labels = np.argmax(trainY, axis=1)
    test_labels = np.argmax(testY, axis=1)
    val_labels = np.argmax(valY, axis=1)
    n_train_labels = train_labels.shape[0]
    n_val_labels = val_labels.shape[0]
    n_test_labels = test_labels.shape[0]

    if debug is True:
        print("Training Labels Size: ", n_train_labels)
        print("Validation Labels Size: ", n_val_labels)
        print("Testing Labels Size: ", n_test_labels)

    train_batches = zip(range(0, n_train - batch_size, batch_size), range(batch_size, n_train, batch_size))
    val_batches = zip(range(0, n_val - batch_size, batch_size), range(batch_size, n_val, batch_size))
    test_batches = zip(range(0, n_test - batch_size, batch_size), range(batch_size, n_test, batch_size))

    return [trainS, trainQ, trainY], list(train_batches), [valS, valQ, valY], list(val_batches), [testS, testQ, testY], \
           list(test_batches)


def vectorize_data(data, word_idx, sentence_size, memory_size):
    '''
    Vectorize stories and queries.

    If a sentence length < sentence_size, the sentence will be padded with 0's.

    If a story length < memory_size, the story will be padded with empty memories.
    Empty memories are 1-D arrays of length sentence_size filled with 0's.

    The answer array is returned as a one-hot encoding.
    '''
    S = []
    Q = []
    A = []
    for story, query, answer in data:
        lq = max(0, sentence_size - len(query))
        q = [word_idx[w] for w in query] + [0] * lq

        ss = []
        for i, sentence in enumerate(story, 1):
            ls = max(0, sentence_size - len(sentence))
            ss.append([word_idx[w] for w in sentence] + [0] * ls)

        if len(ss) > memory_size:

            # Use Jaccard similarity to determine the most relevant sentences
            q_words = (q)
            least_like_q = sorted(ss, key=functools.cmp_to_key(
                lambda x, y: jaccard_similarity_score((x), q_words) < jaccard_similarity_score((y), q_words)))[
                           :len(ss) - memory_size]
            for sent in least_like_q:
                # Remove the first occurrence of sent. A list comprehension as in [sent for sent in ss if sent not in least_like_q]
                # should not be used, as it would remove multiple occurrences of the same sentence, some of which might actually make the cutoff.
                ss.remove(sent)
        else:
            # pad to memory_size
            lm = max(0, memory_size - len(ss))
            for _ in range(lm):
                ss.append([0] * sentence_size)

        y = np.zeros(len(word_idx) + 1)  # 0 is reserved for nil word
        for a in answer:
            y[word_idx[a]] = 1

        S.append(ss)
        Q.append(q)
        A.append(y)
    return np.array(S), np.array(Q), np.array(A)


def generate_batches(batches_tr, batches_v, batches_te, train, val, test):
    train_batches = get_batch_from_batch_list(batches_tr, train)
    val_batches = get_batch_from_batch_list(batches_v, val)
    test_batches = get_batch_from_batch_list(batches_te, test)

    return train_batches, val_batches, test_batches


def get_batch_from_batch_list(batches_tr, train):
    trainS, trainQ, trainA = train
    trainA, trainQ, trainS = extract_tensors(trainA, trainQ, trainS)
    train_batches = []
    train_batches = construct_s_q_a_batch(batches_tr, train_batches, trainS, trainQ, trainA)
    return train_batches


def extract_tensors(A, Q, S):
    A = torch.from_numpy(A).type(long_tensor_type)
    S = torch.from_numpy(S).type(float_tensor_type)
    Q = np.expand_dims(Q, 1)
    Q = torch.from_numpy(Q).type(long_tensor_type)
    return A, Q, S


def construct_s_q_a_batch(batches, batched_objects, S, Q, A):
    for batch in batches:
        # batches are of form : [(0,2), (2,4),...]
        answer_batch = []
        story_batch = []
        query_batch = []
        for j in range(batch[0], batch[1]):
            answer_batch.append(A[j])
            story_batch.append(S[j])
            query_batch.append(Q[j])
        batched_objects.append([story_batch, query_batch, answer_batch])

    return batched_objects


def process_eval_data(data_dir, task_num, word_idx, sentence_size, vocab_size, memory_size=50, batch_size=2,
                      test_size=.1, debug=True, joint_training=0):
    random_state = None
    data, test, vocab = load_data(data_dir, joint_training, task_num)

    if (joint_training == 0):
        memory_size, sentence_size, vocab_size, word_idx = calculate_parameter_values(data=data, debug=debug,
                                                                                      memory_size=memory_size,
                                                                                      vocab=vocab)
    train_set, train_batches, val_set, val_batches, test_set, test_batches = \
        vectorize_task_data(batch_size, data, debug, memory_size, random_state,
                            sentence_size, test, test_size, word_idx)

    return train_batches, val_batches, test_batches, train_set, val_set, test_set, \
           sentence_size, vocab_size, memory_size, word_idx


def get_position_encoding(batch_size, sentence_size, embedding_size):
    '''
    Position Encoding 
    '''
    encoding = np.ones((embedding_size, sentence_size), dtype=np.float32)
    ls = sentence_size + 1
    le = embedding_size + 1
    for i in range(1, le):
        for j in range(1, ls):
            encoding[i - 1, j - 1] = (i - (le - 1) / 2) * (j - (ls - 1) / 2)
    encoding = 1 + 4 * encoding / embedding_size / sentence_size
    enc_vec = torch.from_numpy(np.transpose(encoding)).type(float_tensor_type)
    lis = []
    for _ in range(batch_size):
        lis.append(enc_vec)
    enc_vec = Variable(torch.stack(lis))
    return enc_vec


def weight_update(name, param):
    update = param.grad
    weight = param.data
    print(name, (torch.norm(update) / torch.norm(weight)).data[0])
