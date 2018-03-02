import random
import pickle

def save(fname, obj):
    with open(fname, 'wb') as f:
        pickle.dump(obj, f)
def load(fname):
    with open(fname, 'rb') as f:
        return pickle.load(f)
		
def create_data(len=100, num=10000):
	datas = []
	indexs = []
	sums = []
	for n in range(num):
		data = []
		for i in range(len):
			x = random.random()
			data.append(x)

		x1 = int(random.random()*len)
		x2 = int(random.random()*len)

		index = []
		for i in range(len):
			if i == x1 or i == x2:
				index.append(1)
			else:
				index.append(0)

		sum = 0.0
		for i in range(len):
			if index[i] == 1:
				sum += data[i]
		
		datas.append(data)
		indexs.append(index)
		sums.append(sum)
	return datas,indexs,sums
if __name__ == "__main__":
	lens = [100,200,400,600]
	for len in lens:
		datas,indexs,sums = create_data(len, 100000)
		save("data/datas_train_%d.pkl"%(len), datas)
		save("data/indexs_train_%d.pkl"%(len), indexs)
		save("data/sums_train_%d.pkl"%(len), sums)

		datas,indexs,sums = create_data(len, 10000)
		save("data/datas_test_%d.pkl"%(len), datas)
		save("data/indexs_test_%d.pkl"%(len), indexs)
		save("data/sums_test_%d.pkl"%(len), sums)

		print("Create data seccuss, the length of data is %d" % len)

		# datas = load("datas_train.pkl")
		# indexs = load("indexs_train.pkl")
		# sums = load("sums_train.pkl")
		# print(datas[0])
		# print(indexs[0])
		# print(sums[0])
	
