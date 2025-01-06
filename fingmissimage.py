import os,cv2
miss=[]
for i in range(10000):
     imageset_dir = os.path.join('./Train/image/')
     img_dir = os.path.join(imageset_dir, "%06d.png"%(i))
     if cv2.imread(img_dir) is None:
          miss.append(i)
print(miss)