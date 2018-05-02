import cv2                 # working with, mainly resizing, images
import numpy as np         # dealing with arrays
import os                  # dealing with directories
from random import shuffle # mixing up or currently ordered data that might lead our network astray in training.
from tqdm import tqdm      # a nice pretty percentage bar for tasks. Thanks to viewer Daniel BA1/4hler for this suggestion
#/home/mahendra/Documents/Study material/Labbased project/dogs_vs_cats
#/home/mahendra/Documents/Study material/Labbased project/dogs_vs_cats
import matplotlib.pyplot as plt  #for plot


TRAIN_DIR = '/home/mahendra/Documents/Study material/Labbased project/Object Classifier/data/train'  #directory of trainnig data
TEST_DIR = '/home/mahendra/Documents/Study material/Labbased project/Object Classifier/data/test'     #directory of testing data
IMG_SIZE = 150
LR = 1e-3

MODEL_NAME = 'objectclassifier-{}-{}.model'.format(LR, '2conv-basic') # just so we remember which saved model is which, sizes must match


def label_img(img):
    word_label = img.split('_')[0]
    # conversion to one-hot array [cat,dog]
    #                            [much cat, no dog]
    if word_label == 'airplane': return [1,0,0,0,0]
    #                             [no cat, very doggo]
    if word_label == 'bathtub': return [0,1,0,0,0]
    if word_label == 'bed': return [0,0,1,0,0]
    if word_label == 'bench': return [0,0,0,1,0]

    elif word_label == 'chair': return [0,0,0,0,1]


def create_train_data():
    training_data = []
    for img in tqdm(os.listdir(TRAIN_DIR)):
        label = label_img(img)
        path = os.path.join(TRAIN_DIR,img)
        for img1 in tqdm(os.listdir(path)):
            path1= os.path.join(path,img1)
            img1 = cv2.imread(path1,cv2.IMREAD_GRAYSCALE)
            img1 = cv2.resize(img1, (IMG_SIZE,IMG_SIZE))
            training_data.append([np.array(img1),np.array(label)])
    shuffle(training_data)
    np.save('train_data.npy', training_data)
    return training_data


def process_test_data():
    testing_data = []
    for img in tqdm(os.listdir(TEST_DIR)):
        label = label_img(img)
        path = os.path.join(TEST_DIR,img)
        for img1 in tqdm(os.listdir(path)):
            path1= os.path.join(path,img1)
            img1 = cv2.imread(path1,cv2.IMREAD_GRAYSCALE)
            img1 = cv2.resize(img1, (IMG_SIZE,IMG_SIZE))
            testing_data.append([np.array(img1),np.array(label)])
    shuffle(testing_data)
    np.save('test_data.npy', testing_data)
    return testing_data

if os.path.exists('/home/mahendra/Documents/Study material/Labbased project/Object Classifier/train_data.npy'):
    train_data = np.load('train_data.npy')
else:
    train_data = create_train_data()

#train_data = create_train_data()

# If you have already created the dataset:
#train_data = np.load('train_data.npy')

################*** Model definition **********##################

import tensorflow as tf
import tflearn
from tflearn.layers.conv import conv_2d, max_pool_2d
from tflearn.layers.core import input_data, dropout, fully_connected
from tflearn.layers.estimator import regression

tf.reset_default_graph()
convnet = input_data(shape=[None, IMG_SIZE, IMG_SIZE, 1], name='input')

convnet = conv_2d(convnet, 32, 5, activation='relu')
convnet = max_pool_2d(convnet, 5)

convnet = conv_2d(convnet, 64, 5, activation='relu')
convnet = max_pool_2d(convnet, 5)

convnet = conv_2d(convnet, 128, 5, activation='relu')
convnet = max_pool_2d(convnet, 5)

convnet = conv_2d(convnet, 64, 5, activation='relu')
convnet = max_pool_2d(convnet, 5)

convnet = conv_2d(convnet, 32, 5, activation='relu')
convnet = max_pool_2d(convnet, 5)

convnet = fully_connected(convnet, 1024, activation='relu')
convnet = dropout(convnet, 0.8)

convnet = fully_connected(convnet, 5, activation='softmax')
convnet = regression(convnet, optimizer='adam', learning_rate=LR, loss='categorical_crossentropy', name='targets')

model = tflearn.DNN(convnet, tensorboard_dir='log')


#C:/Users/H/Desktop/KaggleDogsvsCats
if os.path.exists('/home/mahendra/Documents/Study material/Labbased project/Object Classifier/{}.meta'.format(MODEL_NAME)):
    model.load(MODEL_NAME)
    print('model loaded!')
#else:
    train = train_data[:-1000]
    test = train_data[-500:]

    X = np.array([i[0] for i in train]).reshape(-1,IMG_SIZE,IMG_SIZE,1)
    #X=[i[0] for i in train]
    Y = [i[1] for i in train]

    test_x = np.array([i[0] for i in test]).reshape(-1,IMG_SIZE,IMG_SIZE,1)
    test_y = [i[1] for i in test]
    for i in test_y:
        print i

    model.fit({'input': X}, {'targets': Y}, n_epoch=10, validation_set=({'input': test_x}, {'targets': test_y}), 
        snapshot_step=500, show_metric=True, run_id=MODEL_NAME)

    model.save(MODEL_NAME)

#############Testing data ###################################################

# if you need to create the data:
if os.path.exists('/home/mahendra/Documents/Study material/Labbased project/Object Classifier/test_data.npy'):
    test_data = np.load('test_data.npy')
else:
    test_data = process_test_data()


# if you already have some saved:
#test_data = np.load('test_data.npy')



fig=plt.figure()

for num,data in enumerate(test_data[100:125]):
    # cat: [1,0]
    # dog: [0,1]
    
    img_num = data[1]
    img_data = data[0]
    
    y = fig.add_subplot(3,9,num+1)
    orig = img_data
    data = img_data.reshape(IMG_SIZE,IMG_SIZE,1)
    model_out = model.predict([data])[0]
    #model_out = MODEL_NAME.predict(data)
    print model_out
    print np.argmax(model_out)
    if np.argmax(model_out) == 0: 
        str_label='airplane'
        print "airplane\n"
    if np.argmax(model_out) == 1: 
        str_label='bathtub'
        print "bathtub\n"
    if np.argmax(model_out) == 2:  
        str_label='bed'
        print "bed\n"
    if np.argmax(model_out) == 3:  
        str_label='bench'
        print "bench\n"
    if np.argmax(model_out) == 4:  
        str_label='chair'
        print "chair\n"
        
    y.imshow(orig,cmap='gray')
    plt.title(str_label)
    y.axes.get_xaxis().set_visible(False)
    y.axes.get_yaxis().set_visible(False)

plt.show()
