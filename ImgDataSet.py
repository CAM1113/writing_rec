from torch.utils.data import Dataset

# 训练集文件
from FileReader import decode_idx3_ubyte, decode_idx1_ubyte






class ImgDataSet(Dataset):

    def __init__(self, images, labels):
        self.images = decode_idx3_ubyte(images)
        self.labels = decode_idx1_ubyte(labels)

    def __getitem__(self, index):
        return self.images[index]/255, self.labels[index]

    def __len__(self):
        return self.images.shape[0]
