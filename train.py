from __future__ import division
import os,time,cv2
from tqdm import tqdm
import numpy as np
import torch.utils.data as torch_data
from torch.utils.data import DataLoader
# 用于记录日志
import logging
import torch
import torch.nn as nn
import torch.nn.functional as F
import time
import torch.optim as optim
from loss import *
from datasets import G1G2Dataset
import torchvision.transforms as transforms
from models import *
import csv
#os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True'
max_epoch_num = 30
max_test_num = 12000  
mini_batch_size = 10
NO_USE_NORMALIZATION = 0     
# is_training = True
is_training = True
max_patch_num = 140000
trainImageSize = 128
ReadColorImage=1 
isJointTrain = False
lambda1 = 100
lambda2 = 10

##################################################################################
def create_logger(log_file):
    # 定义好logger
    log_format = '%(asctime)s  %(levelname)5s  %(message)s'
    logging.basicConfig(level=logging.INFO, format=log_format, filename=log_file)
    #console = logging.StreamHandler() # 日志输出到流
    #console.setLevel(logging.INFO) # 日志等级
    #console.setFormatter(logging.Formatter(log_format)) # 设置日志格式
    #logging.getLogger(__name__).addHandler(console)
    return logging.getLogger(__name__)

def save_checkpoint(state, filename='checkpoint'):
    filename = '{}.pth'.format(filename)
    torch.save(state, filename)

def checkpoint_state(model=None, optimizer=None, epoch=None, it=None):
    optim_state = optimizer.state_dict() if optimizer is not None else None
    if model is not None:
        if isinstance(model, torch.nn.DataParallel):
            model_state = model.module.state_dict()
        else:
            model_state = model.state_dict()
    else:
        model_state = None

    return {'epoch': epoch, 'it': it, 'model_state': model_state, 'optimizer_state': optim_state}

if __name__ == '__main__':
    # 保存输出的总路径
    # 检查CUDA是否可用
    if torch.cuda.is_available():
        # 获取GPU数量
        n_gpu = torch.cuda.device_count()
        print(f"Number of GPUs available: {n_gpu}")


    root_result_dir = os.path.join('pytorch_outputs') 
    os.makedirs(root_result_dir, exist_ok=True)

    # 当前时间，日志文件的后缀
    time_suffix = time.strftime('%Y-%m-%d_%H:%M:%S',time.localtime(time.time()))
    # 日志文件
    log_file = os.path.join(root_result_dir, 'log_train_g1g2_{}.txt'.format(time_suffix.replace(':', '_')))
    logger = create_logger(log_file)

    # 定义dataset
    trainsplit = G1G2Dataset(ReadColorImage, mode='train')
    trainset = DataLoader(trainsplit, batch_size=mini_batch_size, pin_memory=True,
                              num_workers=4, shuffle=True, drop_last=True)
    testsplit = G1G2Dataset(ReadColorImage, mode='test', test_data= 'MDvsFA')
    testset = DataLoader(testsplit, batch_size=1, pin_memory=True,
                              num_workers=4, shuffle=False, drop_last=True)
    
    # 定义3个Model
    g1 = G1_8()
    g1.cuda()
    g2 = G2_64()
    g2.cuda()
    dis = discriminator()
    dis.cuda()

    # 定义3个优化器
    optimizer_g1 = optim.Adam(g1.parameters(), lr=1e-4, betas=(0.5,0.999))
    optimizer_g2 = optim.Adam(g2.parameters(), lr=1e-4, betas=(0.5,0.999))
    optimizer_d = optim.Adam(dis.parameters(), lr=1e-5, betas=(0.5,0.999))

    # 定义loss
    loss1 = nn.BCEWithLogitsLoss()

    it = 0
    best_epoch = 0
    best_val_F1 = 0
    best_val_g1_F1 = 0
    best_val_g2_F1 = 0
    recall_list1=[]
    recall_list2=[]
    recall_list3=[]
    prec_list1=[]
    prec_list2=[]
    prec_list3=[]
    F1score_list1=[]
    F1score_list2=[]
    F1score_list3=[]
    for epoch in tqdm(range(0, max_epoch_num), desc='epoch', position=1):
        # 调整学习率
        if (epoch+1) % 10 == 0:
            for p in optimizer_g1.param_groups:
                p['lr'] *= 0.2
            for q in optimizer_g2.param_groups:
                q['lr'] *= 0.2
            for r in optimizer_g2.param_groups:
                r['lr'] *= 0.2
        # 训练一个周期
        logger.info('Now we are training epoch {}!'.format(epoch+1))
        total_it_per_epoch = len(trainset)
        
        for bt_idx, data in enumerate(tqdm(trainset, desc='batch', position=0, colour='green')):
            # 训练一个batch
            torch.cuda.empty_cache() # 释放之前占用的缓存
            it = it + 1
            logger.info('current iteration {}/{}, epoch {}/{}, total iteration: {}, g1 lr: {}, g2 lr: {}, Dis lr: {}'.format(
            bt_idx+1, total_it_per_epoch, epoch+1, max_epoch_num, it, float(optimizer_g1.param_groups[0]['lr']), 
            float(optimizer_g2.param_groups[0]['lr']), float(optimizer_d.param_groups[0]['lr'])))

            # 先训练判别器
            dis.train() 
            g1.eval()
            g2.eval()
            optimizer_g1.zero_grad()
            optimizer_g2.zero_grad()
            optimizer_d.zero_grad()

            # 将输入输出放到cuda上
            input_images, output_images = data['input_images'], data['output_images']   # [B, 1, 128, 128]
            input_images = input_images.cuda(non_blocking=True).float()
            output_images = output_images.cuda(non_blocking=True).float()

            g1_out = g1(input_images) # [B, 1, 128, 128]
            g1_out = torch.clamp(g1_out, 0.0, 1.0)

            g2_out = g2(input_images) # [B, 1, 128, 128]
            g2_out = torch.clamp(g2_out, 0.0, 1.0)

            pos1 = torch.cat([input_images, 2 * output_images - 1], dim=1) # [B, 2, 128, 128]
            neg1 = torch.cat([input_images, 2 * g1_out - 1], dim=1) # [B, 2, 128, 128]  
            neg2 = torch.cat([input_images, 2 * g2_out - 1], dim=1) # [B, 2, 128, 128] 
            disc_input  = torch.cat([pos1, neg1, neg2], dim=0) # [3*B, 2, 128, 128]

            logits_real, logits_fake1, logits_fake2, Lgc = dis(disc_input) # [B, 3] [B, 3] [B, 3] [B, 1]

            const1 = torch.ones(mini_batch_size, 1).cuda(non_blocking=True).float()
            const0 = torch.zeros(mini_batch_size, 1).cuda(non_blocking=True).float()

            gen_gt  = torch.cat([const1, const0, const0], dim=1)
            gen_gt1 = torch.cat([const0, const1, const0], dim=1)
            gen_gt2 = torch.cat([const0, const0, const1], dim=1)

            ES0 = torch.mean(loss1(logits_real, gen_gt)) 
            ES1 = torch.mean(loss1(logits_fake1, gen_gt1)) 
            ES2 = torch.mean(loss1(logits_fake2, gen_gt2))

            disc_loss = ES0 + ES1 + ES2
            #logger.info(" discriminator loss is {}".format(disc_loss))
            disc_loss.backward()  # 将误差反向传播
            optimizer_d.step()  # 更新参数

            # 再训练g1
            dis.eval() 
            g1.train()
            g2.eval()
            optimizer_g1.zero_grad()
            optimizer_g2.zero_grad()
            optimizer_d.zero_grad()

            g1_out = g1(input_images) # [B, 1, 128, 128]
            g1_out = torch.clamp(g1_out, 0.0, 1.0)
            MD1 = torch.mean(torch.mul(torch.pow(g1_out - output_images, 2), output_images))
            FA1 = torch.mean(torch.mul(torch.pow(g1_out - output_images, 2), 1 - output_images))
            MF_loss1 = lambda1 * MD1 + FA1 

            g2_out = g2(input_images) # [B, 1, 128, 128]
            g2_out = torch.clamp(g2_out, 0.0, 1.0)

            pos1 = torch.cat([input_images, 2 * output_images - 1], dim=1) # [B, 2, 128, 128]
            neg1 = torch.cat([input_images, 2 * g1_out - 1], dim=1) # [B, 2, 128, 128]  
            neg2 = torch.cat([input_images, 2 * g2_out - 1], dim=1) # [B, 2, 128, 128] 
            disc_input  = torch.cat([pos1, neg1, neg2], dim=0) # [3*B, 2, 128, 128]

            logits_real, logits_fake1, logits_fake2, Lgc = dis(disc_input) # [B, 3] [B, 3] [B, 3] [B, 1]

            const1 = torch.ones(mini_batch_size, 1).cuda(non_blocking=True).float()
            const0 = torch.zeros(mini_batch_size, 1).cuda(non_blocking=True).float()

            gen_gt  = torch.cat([const1, const0, const0], dim=1)
            gen_gt1 = torch.cat([const0, const1, const0], dim=1)
            gen_gt2 = torch.cat([const0, const0, const1], dim=1)


            gen_adv_loss1 = torch.mean(loss1(logits_fake1, gen_gt))
            gen_loss1  = 100*MF_loss1 + 10*gen_adv_loss1 + 1*Lgc
            #logger.info(" g1 loss is {}".format(gen_loss1))

            gen_loss1.backward()  # 将误差反向传播
            optimizer_g1.step()  # 更新参数

            # 再训练g2
            dis.eval() 
            g1.eval()
            g2.train()
            optimizer_g1.zero_grad()
            optimizer_g2.zero_grad()
            optimizer_d.zero_grad()

            g1_out = g1(input_images) # [B, 1, 128, 128]
            g1_out = torch.clamp(g1_out, 0.0, 1.0)

            g2_out = g2(input_images) # [B, 1, 128, 128]
            g2_out = torch.clamp(g2_out, 0.0, 1.0)
            MD2 = torch.mean(torch.mul(torch.pow(g2_out - output_images, 2), output_images))
            FA2 = torch.mean(torch.mul(torch.pow(g2_out - output_images, 2), 1 - output_images))
            MF_loss2 = MD2 + lambda2 * FA2

            pos1 = torch.cat([input_images, 2 * output_images - 1], dim=1) # [B, 2, 128, 128]
            neg1 = torch.cat([input_images, 2 * g1_out - 1], dim=1) # [B, 2, 128, 128]  
            neg2 = torch.cat([input_images, 2 * g2_out - 1], dim=1) # [B, 2, 128, 128] 
            disc_input  = torch.cat([pos1, neg1, neg2], dim=0) # [3*B, 2, 128, 128]

            logits_real, logits_fake1, logits_fake2, Lgc = dis(disc_input) # [B, 3] [B, 3] [B, 3] [B, 1]

            const1 = torch.ones(mini_batch_size, 1).cuda(non_blocking=True).float()
            const0 = torch.zeros(mini_batch_size, 1).cuda(non_blocking=True).float()

            gen_gt  = torch.cat([const1, const0, const0], dim=1)
            gen_gt1 = torch.cat([const0, const1, const0], dim=1)
            gen_gt2 = torch.cat([const0, const0, const1], dim=1)


            gen_adv_loss2 = torch.mean(loss1(logits_fake2, gen_gt)) 
            gen_loss2  = 100*MF_loss2 + 10*gen_adv_loss2 + 1*Lgc
            #logger.info(" g2 loss is {}".format(gen_loss2))

            gen_loss2.backward()  # 将误差反向传播
            optimizer_g2.step()  # 更新参数

            if (bt_idx+1) % 10 == 0:
                # 在测试集上测试
                sum_val_loss_g1 = 0
                sum_val_prec_g1 = 0 
                sum_val_recall_g1 = 0
                sumRealTarN_g1 = 0
                sumDetTarN_g1 = 0
                sum_val_F1_g1 = 0
                g1_time = 0

                sum_val_loss_g2 = 0
                sum_val_prec_g2 = 0 
                sum_val_recall_g2 = 0
                sumRealTarN_g2 = 0
                sumDetTarN_g2 = 0
                sum_val_F1_g2 = 0
                g2_time = 0

                sum_val_loss_g3 = 0
                sum_val_prec_g3 = 0 
                sum_val_recall_g3 = 0
                sumRealTarN_g3 = 0
                sumDetTarN_g3 = 0
                sum_val_F1_g3 = 0

                for bt_idx_test, data in enumerate(testset):
                    g1.eval()
                    g2.eval()
                    dis.eval()
                    optimizer_g1.zero_grad()
                    optimizer_g2.zero_grad()
                    optimizer_d.zero_grad()

                    # 将输入输出放到cuda上
                    input_images, output_images = data['input_images'], data['output_images']   # [B, 1, 128, 128]
                    input_images = input_images.cuda(non_blocking=True).float()
                    output_images = output_images.cuda(non_blocking=True).float()

                   
                    g1_out = g1(input_images) # [B, 1, 128, 128]
                    g1_out = torch.clamp(g1_out, 0.0, 1.0)

  
                    g2_out = g2(input_images) # [B, 1, 128, 128]
                    g2_out = torch.clamp(g2_out, 0.0, 1.0)

                    g3_out = (g1_out + g2_out) / 2 # 取均值的方式进行融合

                    output_images = output_images.cpu().numpy()
                    g1_out = g1_out.detach().cpu().numpy()
                    g2_out = g2_out.detach().cpu().numpy()
                    g3_out = g3_out.detach().cpu().numpy()
                    # 算g1
                    val_loss_g1 = np.mean(np.square(g1_out - output_images))
                    sum_val_loss_g1 += val_loss_g1
                    val_prec_g1, val_recall_g1, val_F1_g1 = calculateF1Measure(g1_out, output_images, 0.5)
                    sum_val_F1_g1 += val_F1_g1
                    sum_val_prec_g1 += val_prec_g1
                    sum_val_recall_g1 += val_recall_g1
                    

                    # 算g2
                    val_loss_g2 = np.mean(np.square(g2_out - output_images))
                    sum_val_loss_g2 += val_loss_g2
                    val_prec_g2, val_recall_g2, val_F1_g2 = calculateF1Measure(g2_out, output_images, 0.5)
                    sum_val_F1_g2 += val_F1_g2
                    sum_val_prec_g2 += val_prec_g2
                    sum_val_recall_g2 += val_recall_g2


                    # 算g3
                    val_loss_g3 = np.mean(np.square(g3_out - output_images))
                    sum_val_loss_g3 += val_loss_g3
                    val_prec_g3, val_recall_g3, val_F1_g3 = calculateF1Measure(g3_out, output_images, 0.5)
                    sum_val_F1_g3 += val_F1_g3
                    sum_val_prec_g3 += val_prec_g3
                    sum_val_recall_g3 += val_recall_g3

                    # 保存图片
                    output_image1 = np.squeeze(g1_out*255.0)#/np.maximum(output_image1.max(),0.0001))
                    output_image2 = np.squeeze(g2_out*255.0)#/np.maximum(output_image2.max(),0.0001))
                    output_image3 = np.squeeze(g3_out*255.0)#/np.maximum(output_image3.max(),0.0001))
                    #cv2.imwrite("%s/%05d_grt.png"%(task,ind),np.uint8(np.squeeze(gt_image*255.0)))
                    cv2.imwrite("pytorch_outputs/results/%05d_G1.png"%(bt_idx_test),np.uint8(output_image1))
                    cv2.imwrite("pytorch_outputs/results/%05d_G2.png"%(bt_idx_test),np.uint8(output_image2))
                    cv2.imwrite("pytorch_outputs/results/%05d_Res.png"%(bt_idx_test),np.uint8(output_image3))    
                                
                
                logger.info("======================== g1 results ============================")
                avg_val_loss_g1 = sum_val_loss_g1/100
                avg_val_prec_g1  = sum_val_prec_g1/100
                avg_val_recall_g1 = sum_val_recall_g1/100
                avg_val_F1_g1 = sum_val_F1_g1/100
                F1score_list1.append(avg_val_F1_g1)
                prec_list1.append(avg_val_prec_g1)
                recall_list1.append(avg_val_recall_g1)

                
                logger.info("================val_L2_loss is %f"% (avg_val_loss_g1))
                logger.info("================prec_rate is %f"% (avg_val_prec_g1))
                logger.info("================recall_rate is %f"% (avg_val_recall_g1))
                logger.info("================F1 measure is %f"% (avg_val_F1_g1))
                logger.info("g1 time is {}".format(g1_time))

                logger.info("======================== g2 results ============================")
                avg_val_loss_g2 = sum_val_loss_g2/100
                avg_val_prec_g2  = sum_val_prec_g2/100
                avg_val_recall_g2 = sum_val_recall_g2/100
                avg_val_F1_g2 = sum_val_F1_g2/100
                F1score_list2.append(avg_val_F1_g2)
                prec_list2.append(avg_val_prec_g2)
                recall_list2.append(avg_val_recall_g2)

                logger.info("================val_L2_loss is %f"% (avg_val_loss_g2))
                logger.info("================prec_rate is %f"% (avg_val_prec_g2))
                logger.info("================recall_rate is %f"% (avg_val_recall_g2))
                logger.info("================F1 measure is %f"% (avg_val_F1_g2))
                logger.info("g2 time is {}".format(g2_time))
                


                logger.info("======================== g3 results ============================")
                avg_val_loss_g3 = sum_val_loss_g3/100
                avg_val_prec_g3  = sum_val_prec_g3/100
                avg_val_recall_g3 = sum_val_recall_g3/100
                avg_val_F1_g3 = sum_val_F1_g3/100
                F1score_list3.append(avg_val_F1_g3)
                prec_list3.append(avg_val_prec_g3)
                recall_list3.append(avg_val_recall_g3)

                logger.info("================val_L2_loss is %f"% (avg_val_loss_g3))
                logger.info("================prec_rate is %f"% (avg_val_prec_g3))
                logger.info("================recall_rate is %f"% (avg_val_recall_g3))
                logger.info("================F1 measure is %f"% (avg_val_F1_g3))
                
                if epoch >=9:
                    if (bt_idx+1) % 100 == 0:
                        ############# save model
                            ckpt_name1 = os.path.join(root_result_dir, 'models/g1_epoch_{}_batch_{}'.format(epoch+1, bt_idx+1))
                            ckpt_name2 = os.path.join(root_result_dir, 'models/g2_epoch_{}_batch_{}'.format(epoch+1, bt_idx+1))
                            ckpt_name3 = os.path.join(root_result_dir, 'models/dis_epoch_{}_batch_{}'.format(epoch+1, bt_idx+1))
                            save_checkpoint(checkpoint_state(g1, optimizer_g1, epoch+1, it), filename=ckpt_name1)
                            save_checkpoint(checkpoint_state(g2, optimizer_g2, epoch+1, it), filename=ckpt_name2)
                            save_checkpoint(checkpoint_state(dis, optimizer_d, epoch+1, it), filename=ckpt_name3)

                # save best F1 model
                if avg_val_F1_g3 > best_val_F1:
                    best_val_F1 = avg_val_F1_g3
                    best_epoch = epoch
                    print('')
                    print("best epoch: %d, best F1: %.4f"%(best_epoch, best_val_F1))
                    print('')
                    ############# save model
                    ckpt_name1 = os.path.join(root_result_dir, 'models/g1_epoch_{}_batch_{}'.format(epoch+1, bt_idx+1))
                    ckpt_name2 = os.path.join(root_result_dir, 'models/g2_epoch_{}_batch_{}'.format(epoch+1, bt_idx+1))
                    ckpt_name3 = os.path.join(root_result_dir, 'models/dis_epoch_{}_batch_{}'.format(epoch+1, bt_idx+1))

                    save_checkpoint(checkpoint_state(g1, optimizer_g1, epoch+1, it), filename=ckpt_name1)
                    save_checkpoint(checkpoint_state(g2, optimizer_g2, epoch+1, it), filename=ckpt_name2)
                    save_checkpoint(checkpoint_state(dis, optimizer_d, epoch+1, it), filename=ckpt_name3)

                if avg_val_F1_g1 > best_val_g1_F1:
                    best_val_g1_F1 = avg_val_F1_g1
                if avg_val_F1_g2 > best_val_g2_F1:
                    best_val_g2_F1 = avg_val_F1_g2

print('')
print('')
print("best epoch: %d, best F1: %.4f"%(best_epoch, best_val_F1))
plotloss(F1score_list1, recall_list1, prec_list1, 1)
plotloss(F1score_list2, recall_list2, prec_list2, 2)
plotloss(F1score_list3, recall_list3, prec_list3, 3)

# 打开一个文件用于写入CSV
with open('saveF1.csv', 'w', newline='') as file:
    writer = csv.writer(file)
    
    # 遍历列表，每次从每个列表中取出一个元素，组成一行写入文件
    for item1, item2, item3, item4, item5, item6 in zip(F1score_list1, F1score_list2, F1score_list3, recall_list1, recall_list2, recall_list3, prec_list1,prec_list2,prec_list3):
        writer.writerow([item1, item2, item3, item4, item5, item6])

                


