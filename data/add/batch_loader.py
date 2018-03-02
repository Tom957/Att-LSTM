import pickle
import os
import numpy as np

def save(fname, obj):
    with open(fname, 'wb') as f:
        pickle.dump(obj, f)
def load(fname):
    with open(fname, 'rb') as f:
        return pickle.load(f)

class BatchLoader():
    def __init__(self, batch_size, step_size, is_test=False):
        self.data_name = 'add'
        self.batch_size = batch_size
        self.step_size = step_size
        if is_test:
            path = os.path.join(os.getcwd(),'data')
        else:
            path = os.path.join(os.getcwd(), 'data', self.data_name, 'data')

        datas_train_fname = os.path.join(path, 'datas_train_%d.pkl'%(step_size))
        indexs_train_fname = os.path.join(path, 'indexs_train_%d.pkl'%(step_size))
        sums_train_fname = os.path.join(path,  'sums_train_%d.pkl'%(step_size))
        datas_test_fname = os.path.join(path,  'datas_test_%d.pkl'%(step_size))
        indexs_test_fname = os.path.join(path, 'indexs_test_%d.pkl'%(step_size))
        sums_test_fname = os.path.join(path, 'sums_test_%d.pkl'%(step_size))

        self.datas_train = load(datas_train_fname)
        self.indexs_train = load(indexs_train_fname)
        self.sums_train = load(sums_train_fname)

        self.datas_test = load(datas_test_fname)
        self.indexs_test = load(indexs_test_fname)
        self.sums_test = load(sums_test_fname)

        self.train_ptr = 0
        self.test_ptr = 0

        self.train_num = int(len(self.datas_train) / batch_size)
        self.test_num = int(len(self.datas_test) / batch_size)

    def next_test_batch(self):
        data = self.datas_test[self.test_ptr*self.batch_size : (self.test_ptr+1)*self.batch_size]
        index = self.indexs_test[self.test_ptr*self.batch_size : (self.test_ptr+1)*self.batch_size]
        sums = self.sums_test[self.test_ptr*self.batch_size : (self.test_ptr+1)*self.batch_size]
        sums = np.array(sums)
        sums = sums.reshape((self.batch_size,1))
        self.test_ptr += 1
        self.test_ptr %= self.test_num
        inputs = []
        for i in range(self.batch_size):
            input = np.zeros([self.step_size, 2])
            for j in range(self.step_size):
                input[j][0] = data[i][j]
                input[j][1] = index[i][j]
            inputs.append(input)
        inputs = np.array(inputs)
        return inputs, sums


    def next_train_batch(self):
        data = self.datas_train[self.train_ptr * self.batch_size: (self.train_ptr + 1) * self.batch_size]
        index = self.indexs_train[self.train_ptr * self.batch_size: (self.train_ptr + 1) * self.batch_size]
        sums = self.sums_train[self.train_ptr * self.batch_size: (self.train_ptr + 1) * self.batch_size]
        sums = np.array(sums)
        sums = sums.reshape((self.batch_size,1))
        self.train_ptr += 1
        self.train_ptr %= self.train_num
        inputs = []
        for i in range(self.batch_size):
            input = np.zeros([self.step_size , 2])
            for j in range(self.step_size):
                input[j][0] = data[i][j]
                input[j][1] = index[i][j]
            inputs.append(input)
        inputs = np.array(inputs)
        return inputs, sums
		
if __name__ == "__main__":
    b = BatchLoader(20, 400, True)
    print(b.train_num)
    print(b.test_num)
    for i in range(b.test_num):
        print("i->>>"+str(i))
        batch_xs, batch_ys = b.next_test_batch()

        print(batch_xs.shape)
        for i, (x, y) in enumerate(zip(batch_xs, batch_ys)):
            # print(x)
            for j, a in enumerate(x):
                if(a[1] == 1):
                    print(str(j)+"->>>"+str(a[0]))
            print(y)
            print()


