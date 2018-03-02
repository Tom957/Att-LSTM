import numpy as np
import pickle
import os

def save(fname, obj):
    with open(fname, 'wb') as f:
        pickle.dump(obj, f)
def load(fname):
    with open(fname, 'rb') as f:
        return pickle.load(f)

class BatchLoader():
    def __init__(self, config, is_local=False):
        self.data_name = 'trec'
        self.y_dict = {'LOC': [1,0,0,0,0,0],
                       'ABBR': [0,1,0,0,0,0],
                       'DESC': [0,0,1,0,0,0],
                       'HUM': [0,0,0,1,0,0],
                       'ENTY': [0,0,0,0,1,0],
                       'NUM': [0,0,0,0,0,1]}
        self.class_num = len(self.y_dict)
        self.batch_size = config.batch_size
        max_sent_len_my_set = config.max_sent_len
        max_word_len_my_set = config.max_word_len
        if is_local:
            path = ''
        else:
            path = os.path.join(os.getcwd(), "data", self.data_name)
        vocab_file_name = os.path.join(path, 'vocab.pkl')
        train_file_name = os.path.join(path, 'train.txt')
        test_file_name = os.path.join(path, 'test.txt')

        print (vocab_file_name)
        if not os.path.exists(vocab_file_name):
            print("Creating vocab...")
            self.save_vocab_from_data_file(train_file_name, vocab_file_name)
            print("Creating vocab...Done\n")

        print("Loading vocab...")
        vacab = load(vocab_file_name)
        self.idx2word_arr, self.word2idx_dict, self.idx2char_arr, self.char2idx_dict, self.data_len_dict = vacab
        print("Loading vocab...Done\n")

        print("Print Data Info...")
        self.word_vocab_size = len(self.idx2word_arr)
        self.char_vocab_size = len(self.idx2char_arr)
        max_word_len_from_data = self.data_len_dict['max_word_len']
        max_sent_len_from_data = self.data_len_dict['max_sent_len']
        print("\tWord vocab size: %d, Char vocab size: %d" % (self.word_vocab_size, self.char_vocab_size))
        print("\tMax Sent length: %d, Max Word length: %d" % (max_sent_len_from_data, max_word_len_from_data))
        print("Print Data Info...Done\n")

        print("Preperation For Batch...")
        self.max_sent_length = min(max_sent_len_from_data, max_sent_len_my_set)
        self.max_word_length = min(max_word_len_from_data, max_word_len_my_set)

        self.batch_index = 0# random.randint(0, batch_size-1);
        with open(train_file_name) as train_file:
            self.train_text_lines = train_file.readlines()
        with open(test_file_name) as test_file:
            self.test_text_lines = test_file.readlines()
        self.train_text_lines_size = len(self.train_text_lines)
        self.test_text_lines_size = len(self.test_text_lines)
        print('\tself.train_text_lines_size: ', str(self.train_text_lines_size))
        print('\tself.test_text_lines_size: ', str(self.test_text_lines_size))
        print("Preperation For Batch...Done\n")

    def save_vocab_from_data_file(self, data_file_name, vocab_file_name):
        max_word_len = 0
        max_sent_len = 0
        count_split = 0
        vocab_dict = {}
        with open(data_file_name) as f:
            for line in f:
                split = line.split(':')
                line = split[1]
                line = line.replace('\n', '')
                words = line.split()
                max_sent_len = max(max_sent_len, len(words))
                for word in words:
                    max_word_len = max(max_word_len, len(word) + 2)
                    if word not in vocab_dict:
                        vocab_dict[word] = 1
                    else:
                        vocab_dict[word] = vocab_dict[word] +1
                    count_split += 1

        print ("After first pass of data:")
        print ("\tMax sent length is: %d" % max_sent_len)
        print ("\tMax word length is: %d" % max_word_len)
        print ("\tToken count: %d" % (count_split))
        print ("\tVocab count: %d"% len(vocab_dict))

        char2idx_dict = {' ': 0, '{': 1, '}': 2}
        idx2char_array = [' ', '{', '}']
        word2idx_dict = {' ': 0}
        idx2word_array = [' ']

        len_dict = {'max_word_len':max_word_len,'max_sent_len':max_sent_len}

        with open(data_file_name) as f:
            for line in f:
                split = line.split(':')
                line = split[1]
                line = line.replace('\n', '')
                words = line.split()
                for word in words:
                    if word not in word2idx_dict:
                        idx2word_array.append(word)
                        word2idx_dict[word] = len(idx2word_array) - 1

                    for char in word:
                        if char not in char2idx_dict:
                            idx2char_array.append(char)
                            char2idx_dict[char] = len(idx2char_array) - 1

        save(vocab_file_name, [idx2word_array,
                               word2idx_dict,
                               idx2char_array,
                               char2idx_dict,
                               len_dict])
        print('Save vocab file success')

    def get_test_data(self):
        line_arr = self.test_text_lines
        taget_output_arr, \
        words_index_arr, \
        chars_index_arr = self.convert_word_to_index(line_arr=line_arr)

        return taget_output_arr, words_index_arr, chars_index_arr

    def next_batch(self):
        begin_index = self.batch_index
        end_index = self.batch_index + self.batch_size
        isOneEpoch = False
        if end_index >= self.train_text_lines_size:
            line_arr1 = self.train_text_lines[begin_index:self.train_text_lines_size]
            end_index = end_index - self.train_text_lines_size
            line_arr2 = self.train_text_lines[0:end_index]
            line_arr = line_arr1 + line_arr2
            isOneEpoch = True
        else:
            line_arr = self.train_text_lines[begin_index:end_index]

        self.batch_index = end_index
        taget_output_arr, \
        words_index_arr, \
        chars_index_arr = self.convert_word_to_index(line_arr)
        return taget_output_arr, words_index_arr, chars_index_arr, isOneEpoch

    def convert_word_to_index(self, line_arr):
        taget_output_arr = []
        words_index_arr = []
        chars_index_arr = []
        for line in line_arr:
            #print line
            words_index_line = np.zeros(self.max_sent_length, dtype=np.int)
            chars_index_line = np.zeros([self.max_sent_length, self.max_word_length],
                                        dtype=np.int)
            line = line.replace('\n', '')
            label_context = line.split(':')
            label = label_context[0]
            taget_output_arr.append(self.y_dict[label])

            context = label_context[1]
            words = context.split()
            word_count = 0
            for word in words:
                if word_count >= self.max_sent_length:
                    break
                if word in self.word2idx_dict:
                    words_index_line[word_count] = self.word2idx_dict[word]
                else:
                    words_index_line[word_count] = 0
                char_count = 0
                for char in word:
                    if char_count >= self.max_word_length:
                        break
                    if char in self.char2idx_dict:
                        chars_index_line[word_count][char_count] = self.char2idx_dict[char]
                    else:
                        chars_index_line[word_count][char_count] = 0
                    char_count += 1
                word_count += 1

            #self.convert_index_to_word([chars_index_line])
            words_index_arr.append(words_index_line)
            chars_index_arr.append(chars_index_line)

        return taget_output_arr, words_index_arr, chars_index_arr
    def convert_index_to_word(self, chars_index_arr):
        line_size = len(chars_index_arr)
        for i in range(line_size):
            line_str = ''
            line_char = chars_index_arr[i]
            for char_index in line_char:
                last_char = ''
                for char in char_index:
                    cur_char =  self.idx2char_arr[int(char)]
                    if last_char == ' ' and  cur_char == ' ':
                        line_str = line_str + ''
                    else:
                        line_str = line_str + cur_char

                    last_char = self.idx2char_arr[int(char)]
            print(line_str)

def parser_source_data():
    train_file_name = os.path.join(os.getcwd(), 'train.txt')

    texts = []
    labels = []
    with open(train_file_name,'r') as data_file:
        lines = data_file.readlines()
        for line in lines:
            line = line.replace('\n', '')
            splited_line = line.split(':')
            texts.append(splited_line[1])
            labels.append(splited_line[0])
    label_dict = {}
    is_print_data_info = True
    if is_print_data_info == True:
        count = [0,0,0,0,0]
        for i, (label, text) in enumerate( zip(labels, texts) ):
            print("%s--->%s:        %s" % (str(i+1), label, text))
            if label not in label_dict:
                label_dict[label] = 0
            else:
                label_dict[label] += 1
        print(label_dict)



def parse_input():
    import argparse
    parser = argparse.ArgumentParser(description="Parse network configuretion")
    parser.add_argument("--batch_size",type=int, default=100)
    parser.add_argument("--max_sent_len",type=int, default=30)
    parser.add_argument("--max_word_len",type=int, default=10)
    parser.add_argument("--first_unit_size",type=int, default=100)
    parser.add_argument("--secod_unit_size",type=int, default=100)
    parser.add_argument("--char_embed_dim",type=int, default=20)
    parser.add_argument("--epoch_num",type=int, default=800)
    parser.add_argument("--learning_rate",type=float, default=1.0e-3)
    parser.add_argument("--learning_rate_decay",type=float, default=0.95)
    parser.add_argument("--reg_lambda",type=float, default=0)
    parser.add_argument("--clip_norm",type=float, default=1.0)
    parser.add_argument("--device",type=str, default='/gpu:0')
    parser.add_argument("--cell_name",type=str, default="arnn")
    parser.add_argument("--K",type=int, default=6)
    return parser.parse_args()

if __name__ == "__main__":
    config = parse_input();
    loader = BatchLoader(config,True)
    for i in range(10):
        taget_output_arr, words_index_arr, chars_index_arr, isOneEpoch = loader.next_batch()
        loader.convert_index_to_word(chars_index_arr)
        print(taget_output_arr)
        if isOneEpoch == True:
            print('================================\n')
    #parser_source_data()
