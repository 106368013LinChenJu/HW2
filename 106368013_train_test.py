import matplotlib.pyplot as plt
import itertools
from PIL import Image
from sklearn.metrics import confusion_matrix

import os,sys
import cv2
import numpy as np
from sklearn.model_selection import train_test_split
from keras.models import Sequential, Model, load_model
from keras import applications
from keras.layers import *
from keras.callbacks import *
from keras.utils import np_utils
from keras.optimizers import SGD, Adam
from keras.preprocessing.image import ImageDataGenerator

epochs=150
batch_size=256

images=[]
labels= []
listdir= []

def read_images_labels(path,i):
    for file in os.listdir(path):
        abs_path = os.path.abspath(os.path.join(path, file))   # abs_path =  C:\\XXX\XXX\ + train\XXX\  ||  +(XXX).jpg 
        if os.path.isdir(abs_path):
            i+=1                                               # 1- 20
            temp = os.path.split(abs_path)[-1]                 # C:\\XXX\XXX\ + train\XXX\ >> XXX
            listdir.append(temp)                               # stack file path
            read_images_labels(abs_path,i)                     # read_images_labels(C:\\XXX\XXX\ + train\XXX\)
            amount = int(len(os.listdir(path)))              # train\ file amount
            sys.stdout.write('\r'+'>'*(i)+' '*(amount-i)+'[%s%%]'%(i*100/amount)+temp) #Loading Bar
        else:  
            if file.endswith('.jpg'):
                image=cv2.resize(cv2.imread(abs_path),(64,64)) # read XXX.jpg resize 64x64
                images.append(image)                           # stack image
                labels.append(i-1)                             # stack labels
    return images, labels ,listdir

def read_main(path):
    images, labels ,listdir = read_images_labels(path,i=0)
    images = np.array(images,dtype=np.float32)/255
    labels = np_utils.to_categorical(labels, num_classes=20)
    np.savetxt('listdir.txt', listdir, delimiter = ' ',fmt="%s")
    return images, labels

images, labels=read_main('characters-20')
X_train, X_test, y_train, y_test = train_test_split(images, labels, test_size=0.1)

print(X_train.shape)
print(y_train.shape)
print(X_test.shape)
print(y_test.shape)

base_model = applications.VGG16(weights='imagenet', include_top=False, input_shape=X_train.shape[1:])

add_model = Sequential()
add_model.add(Flatten(input_shape=base_model.output_shape[1:]))
add_model.add(Dense(256, activation='relu'))
add_model.add(Dense(20, activation='softmax'))
model = Model(inputs=base_model.input, outputs=add_model(base_model.output))

sgd = SGD(lr=0.005, decay=1e-6, momentum=0.9, nesterov=True)
model.compile(loss='categorical_crossentropy', optimizer='adam',metrics=['accuracy'])

datagen = ImageDataGenerator(zoom_range=0.1,width_shift_range=0.05,height_shift_range=0.05,horizontal_flip=True,)
datagen.fit(X_train)

file_name=str(epochs)+'_debug'+str(batch_size)
TB=TensorBoard(log_dir='logs/'+file_name)

history = model.fit_generator(datagen.flow(X_train, y_train, batch_size=batch_size),
                    steps_per_epoch=epochs, epochs=epochs,
                    validation_data = (X_test, y_test ), verbose = 1,callbacks=[TB])


plt.plot(history.history['acc'])
plt.plot(history.history['val_acc'])
plt.title('Accuracy')
plt.ylabel('probability')
plt.xlabel('epoch')
plt.legend(['train', 'valid'], loc='upper left')
plt.show()

plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'valid'], loc='upper left')
plt.show()

model.summary()

weight_conv2d_1 = model.layers[1].get_weights()[0][:,:,0,:]
 
col_size = 8
row_size = 8   
filter_index = 0    
fig, ax = plt.subplots(row_size, col_size, figsize=(5,5))

for row in range(0,row_size): 
        for col in range(0,col_size):
            ax[row][col].imshow(weight_conv2d_1[:,:,filter_index],cmap="gray")
            ax[row][col].axis('off')
            filter_index += 1

def plot_confusion_matrix(cm, classes,
                          normalize=False,
                          title='Confusion matrix',
                          cmap=plt.cm.Blues):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    plt.figure(figsize = (10,10))
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=90)
    plt.yticks(tick_marks, classes)

    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]

    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, cm[i, j],
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')


dict_characters = {0: 'abraham_grampa_simpson', 1: 'apu_nahasapeemapetilon', 2: 'bart_simpson', 
        3: 'charles_montgomery_burns', 4: 'chief_wiggum', 5: 'comic_book_guy', 6: 'edna_krabappel', 
        7: 'homer_simpson', 8: 'kent_brockman', 9: 'krusty_the_clown', 10: 'lenny_leonard', 11:'lisa_simpson',
        12: 'marge_simpson', 13: 'mayor_quimby',14:'milhouse_van_houten', 15: 'moe_szyslak', 
        16: 'ned_flanders', 17: 'nelson_muntz', 18: 'principal_skinner', 19: 'sideshow_bob'}
# Predict the values from the validation dataset
Y_pred = model.predict(X_test)
# Convert hot vectors prediction results to list of classes
Y_pred_classes = np.argmax(Y_pred,axis = 1) 
# Convert hot vectors validation observations to list of classes
Y_true = np.argmax(y_test,axis = 1) 
# compute the confusion matrix
confusion_mtx = confusion_matrix(Y_true, Y_pred_classes) 
# plot the confusion matrix
plot_confusion_matrix(confusion_mtx, classes = list(dict_characters.values())) 

model.save(file_name+'debug.h5')
score = model.evaluate(X_test, y_test, verbose=0)
print(score)