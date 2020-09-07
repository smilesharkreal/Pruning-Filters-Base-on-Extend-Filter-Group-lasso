import pickle
import numpy as np
import random
class Cifar():
    def __init__(self,filename):
        self.filename = filename
        splitfile = filename.split("/")
        metaname = "/batches.meta"
        metalabel = b'label_names'
        if splitfile[-1] == "cifar-100-python":
            metaname = "/meta"
            metalabel = b'fine_label_names'
        self.metaname = metaname
        self.metalabel = metalabel
        self.image_size = 32
        self.img_channels = 3

    # 解析数据
    def unpickle(self,filename):
        with open(filename,"rb") as fo:
            dict = pickle.load(fo,encoding='bytes')
        return dict

    # 将数据导入
    def load_data_one(self,file):
        batch = self.unpickle(file)
        data = batch[b'data']
        labelname = b'labels'
        if self.metaname== "/meta":
            labelname = b'fine_labels'
        label = batch[labelname]
        print("Loading %s : %d." % (file, len(data)))
        return data,  label

    # 对 label进行处理
    def load_data(self,files,data_dir, label_count):
        data, labels = self.load_data_one(data_dir + "/" + files[0])
        for f in files[1:]:
            data_n, labels_n = self.load_data_one(data_dir + '/' + f)
            data = np.append(data, data_n, axis=0)
            labels = np.append(labels, labels_n, axis=0)
        labels = np.array([[float(i == label) for i in range(label_count)] for label in labels])
        data = data.reshape([-1, self.img_channels, self.image_size, self.image_size])
        data = data.transpose([0, 2, 3, 1])  # 将图片转置 跟卷积层一致
        return data, labels

    # 数据导入
    def prepare_data(self):
        print("======Loading data======")
        # data_dir = '../Cifar_10/cifar-100-python'
        # image_dim = self.image_size * self.image_size * self.img_channels
        meta = self.unpickle(self.filename + self.metaname)
        label_names = meta[self.metalabel]
        label_count = len(label_names)
        if self.metaname== "/batches.meta":
            train_files = ['data_batch_%d' % d for d in range(1, 6)]
            train_data, train_labels = self.load_data(train_files, self.filename, label_count)
            test_data, test_labels = self.load_data(['test_batch'], self.filename, label_count)
        else:
            train_data, train_labels = self.load_data(['train'],self.filename, label_count)
            test_data, test_labels = self.load_data(['test'] , self.filename, label_count)
        print("Train data:", np.shape(train_data), np.shape(train_labels))
        print("Test data :", np.shape(test_data), np.shape(test_labels))
        print("======Load finished======")
        print("======Shuffling data======")
        indices = np.random.permutation(len(train_data))  # 打乱数组
        train_data = train_data[indices]
        train_labels = train_labels[indices]
        print("======Prepare Finished======")

        return train_data, train_labels, test_data, test_labels

## Z-score 标准化
def data_preprocessing(x_train, x_test):
    x_train = x_train.astype('float32')
    x_test = x_test.astype('float32')

    x_train[:, :, :, 0] = (x_train[:, :, :, 0] - np.mean(x_train[:, :, :, 0])) / np.std(x_train[:, :, :, 0])
    x_train[:, :, :, 1] = (x_train[:, :, :, 1] - np.mean(x_train[:, :, :, 1])) / np.std(x_train[:, :, :, 1])
    x_train[:, :, :, 2] = (x_train[:, :, :, 2] - np.mean(x_train[:, :, :, 2])) / np.std(x_train[:, :, :, 2])

    x_test[:, :, :, 0] = (x_test[:, :, :, 0] - np.mean(x_test[:, :, :, 0])) / np.std(x_test[:, :, :, 0])
    x_test[:, :, :, 1] = (x_test[:, :, :, 1] - np.mean(x_test[:, :, :, 1])) / np.std(x_test[:, :, :, 1])
    x_test[:, :, :, 2] = (x_test[:, :, :, 2] - np.mean(x_test[:, :, :, 2])) / np.std(x_test[:, :, :, 2])
    return x_train, x_test

# 对数据随机左右翻转
def _random_flip_leftright(batch):
    for i in range(len(batch)):
        if bool(random.getrandbits(1)):
            batch[i] = np.fliplr(batch[i])
    return batch

# 随机裁剪
def _random_crop(batch, crop_shape, padding=None):
    oshape = np.shape(batch[0])

    if padding:
        oshape = (oshape[0] + 2 * padding, oshape[1] + 2 * padding)
    new_batch = []
    npad = ((padding, padding), (padding, padding), (0, 0))
    for i in range(len(batch)):
        new_batch.append(batch[i])
        if padding:
            new_batch[i] = np.lib.pad(batch[i], pad_width=npad,
                                      mode='constant', constant_values=0)
        nh = random.randint(0, oshape[0] - crop_shape[0])
        nw = random.randint(0, oshape[1] - crop_shape[1])
        new_batch[i] = new_batch[i][nh:nh + crop_shape[0],
                       nw:nw + crop_shape[1]]
    return new_batch

def data_augmentation(batch):
    batch = _random_flip_leftright(batch)
    batch = _random_crop(batch, [224, 224], 4)
    return batch





