import cv2
import numpy as np
from keras.models import load_model
import pandas as pd


def read_images(path):
    images=[]
    for i in range(990):
        image=cv2.resize(cv2.imread(path+str(i+1)+'.jpg'),(64,64))
        images.append(image)

    images=np.array(images,dtype=np.float32)/255
    return images

def transform(listdir,label,lenSIZE):
    label_str=[]
    for i in range (lenSIZE):
        temp=listdir[label[i]]       
        label_str.append(temp)

    return label_str

images = read_images('test/')
model = load_model('150_debug256debug.h5')

predict = model.predict(images, verbose=1)
#print(predict)
predict_index =np.zeros([990,1])
#for num in range(predict.shape[0]): #將機率轉換成分類名稱索引
#    p=predict[num]
#    list = p.tolist()
#    max_list =  max(list) # 最大值的索引
#    max_index = list.index(max(list)) # 返回最大值位置
#    predict_index[num]=np.array(max_index)

for num in range(predict.shape[0]): #將機率轉換成分類名稱索引
    p=predict[num]
#    list = p.tolist()[0]
    list = p.tolist()
    max_list =  max(list) # 最大值的索引
    max_index = list.index(max(list)) # 返回最大值位置
    predict_index[num]=np.array(max_index)
#    print(p)
predict_index=np.array(predict_index,int)

label_str=transform(np.loadtxt('listdir.txt',dtype='str'),predict_index,images.shape[0])
label_str=np.array(label_str,str)
np.savetxt('prediction3'+'.txt', label_str, delimiter = ' ',fmt="%s")
#pd.DataFrame({"id": list(range(1,len(label_str)+1)),"character": label_str}).to_csv('test_score.csv', index=False, header=True)