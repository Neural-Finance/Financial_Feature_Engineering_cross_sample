#Author: AlexFang, alex.holla@foxmail.com.
import pickle as p
import numpy as np
from PIL import Image


# names = ["airplane","automobile", "bird","cat", "deer", "dog", "frog", "horse", "ship", "truck"]

def load_CIFAR_batch(filename):
    with open(filename, 'rb')as f:
        datadict = p.load(f, encoding='bytes')
        imagedata = datadict[b'data']
        labels = datadict[b'labels']
        imagedata = imagedata.reshape(10000, 3, 32, 32)
        labels = np.array(labels)
        return imagedata, labels


def load_names(filename):
    with open(filename, 'rb')as f:
        namedict = p.load(f, encoding='bytes')
        names = namedict[b'label_names']
        return names


if __name__ == "__main__":

    images, labels = load_CIFAR_batch("./cifar10_data/cifar-10-batches-py/data_batch_1")
    names = load_names("./cifar10_data/cifar-10-batches-py/batches.meta")
    print(images.shape)
    print("正在保存图片……")

    for i in range(3):  # 输出3张图片，利用PIL.Image生成图片
        image = images[i]
        r_array = image[0]
        g_array = image[1]
        b_array = image[2]
        channel_r = Image.fromarray(r_array)
        channel_g = Image.fromarray(g_array)
        channel_b = Image.fromarray(b_array)
        image = Image.merge("RGB", (channel_r, channel_g, channel_b))
        name = "img_" + str(i) + "_" + str(names[labels[i]]) + ".png"
        image.save("./cifar10_data/visualize/" + name, "png")

    print("保存完毕.")
