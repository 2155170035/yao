import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
import os,cv2
def calculateF1Measure(output_image,gt_image,thre):
    output_image = np.squeeze(output_image,axis=1)
    gt_image = np.squeeze(gt_image)
    out_bin = output_image>thre
    gt_bin = gt_image>thre
    recall = np.sum(gt_bin*out_bin)/np.maximum(1,np.sum(gt_bin))
    prec   = np.sum(gt_bin*out_bin)/np.maximum(1,np.sum(out_bin))
    F1 = 2*recall*prec/np.maximum(0.001,recall+prec)
    return prec,recall,F1


def plotloss(F1score_list,recall_list,prec_list,index):
    x_list = list(range(len(F1score_list)))
    plt.plot(x_list, F1score_list)
    plt.xlabel('batches/10')
    plt.ylabel('F1score')
    plt.title('The F1 score shows a gradual improvement every 10 batches')
    plt.show()
    plt.savefig(f'./lossimage/F1score_{index}.png')
    plt.close()
    
    plt.plot(x_list, recall_list)
    plt.xlabel('batches/10')
    plt.ylabel('recall')
    plt.title('The recall_rate shows a gradual improvement every 10 batches')
    plt.show()
    plt.savefig(f'./lossimage/recall_{index}.png')
    plt.close()

    plt.plot(x_list, prec_list)
    plt.xlabel('batches/10')
    plt.ylabel('prec')
    plt.title('The prec_rate shows a gradual improvement every 10 batches')
    plt.show()
    plt.savefig(f'./lossimage/prec_{index}.png')
    plt.close()

if __name__ == '__main__':  
    miss=[]
    for i in range(10000):
        imageset_dir = os.path.join('./Train/image/')
        img_dir = os.path.join(imageset_dir, "%06d.png"%(i))
        if cv2.imread(img_dir) is None:
            miss.append(i)
    print(miss)