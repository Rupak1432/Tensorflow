from glob import glob
from PIL import Image
import numpy as np
from random import shuffle

def load_data():

    apple_scab = glob("/home/rupak/LeanAgri/leaf-disease-plant-village/plantvillage_deeplearning_paper_dataset/color/Apple___Apple_scab/*.JPG")
    label1 = [1 for i in range(len(apple_scab))]

    apple_frogeye_spot = glob("/home/rupak/LeanAgri/leaf-disease-plant-village/plantvillage_deeplearning_paper_dataset/color/Apple_Frogeye_Spot/*.JPG")
    label2 = [2 for i in range(len(apple_frogeye_spot))]

    apple_healthy = glob("/home/rupak/LeanAgri/leaf-disease-plant-village/plantvillage_deeplearning_paper_dataset/color/Apple___healthy/*.JPG")
    label3 = [3 for i in range(len(apple_healthy))]

    filenames = apple_scab + apple_frogeye_spot + apple_healthy
    labels = label1 + label2 + label3

    s = list(zip(filenames,labels))
    shuffle(s)
    filenames,labels = zip(*s)

    train_fn = filenames[0:int(0.6*len(filenames))]
    train_lb = labels[0:int(0.6*len(labels))]

    val_fn = filenames[int(0.6*len(filenames)):int(0.8*len(filenames))]
    val_lb = labels[int(0.6*len(labels)):int(0.8*len(labels))]

    test_fn = filenames[int(0.8*len(filenames)):]
    test_lb = labels[int(0.8*len(labels)):]

    train_ar = np.empty([0,196608])
    val_ar = np.empty([0,196608])
    test_ar = np.empty([0,196608])

    for i in train_fn:
        npar = np.asarray(Image.open(i),dtype=np.float32)/255.
        npar = np.reshape(npar,(1,-1))
        train_ar = np.concatenate((train_ar,npar))

    for i in val_fn:
        npar = np.asarray(Image.open(i), dtype=np.float32)/255.
        npar = np.reshape(npar,(1,-1))
        val_ar = np.concatenate((val_ar,npar))

    for i in test_fn:
        npar = np.asarray(Image.open(i), dtype=np.float32)/255.
        npar = np.reshape(npar,(1,-1))
        test_ar = np.concatenate((test_ar,npar))

    return (train_ar, train_lb), (val_ar, val_lb), (test_ar, test_lb)

