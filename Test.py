import torch
from torch.utils.data.dataloader import DataLoader
import numpy as np
from ImgDataSet import ImgDataSet
from Net import Net
import matplotlib.pyplot as plt
from torch.autograd import Variable

# 测试集文件
test_images_idx3_ubyte_file = './datasets/t10k-images.idx3-ubyte'
# 测试集标签文件
test_labels_idx1_ubyte_file = './datasets/t10k-labels.idx1-ubyte'

imageDataSet = ImgDataSet(test_images_idx3_ubyte_file, test_labels_idx1_ubyte_file)
dataLoader = DataLoader(imageDataSet, batch_size=100, shuffle=True)

net = torch.load("net_1.0_.pt")
net.eval()


def show(index):
    image, label = imageDataSet[index]
    plt.imshow(image)
    plt.show()
    image = torch.from_numpy(image).view(-1, 1, image.shape[0], image.shape[1])
    if torch.cuda.is_available():
        image = image.cuda().float()
    y = net(image)
    y = torch.argmax(y)
    print("预测：{}".format(y))
    print("标签：{}".format(label))


def predict():
    times = 0
    scores = 0
    for index, data in enumerate(dataLoader):
        times += 1
        image, label = data
        if torch.cuda.is_available():
            image = image.cuda()
            label = label.cuda()
        image = image.view(image.shape[0], 1, image.shape[1], image.shape[2])
        image = Variable(image).float()
        label = Variable(label).long()
        y = net(image)

        score = torch.argmax(y, 1)
        score = score == label
        score = score.sum()
        score = score.float() / y.shape[0]
        score = score.cpu().numpy()
        scores += score
    scores = scores / times
    print("测试集上平均尊准确率：{}%".format(scores*100))


# predict()
while True:
    i = int(input("输入index："))
    show(i)
