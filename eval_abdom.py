from datetime import datetime
import json
import numpy as np
import random
import os
import cv2
import time
from basic_blocks import init_weights
from abdom_dataloader import UnalignedDataset as abdom_dataset
import torch.nn as nn
import torch
import itertools
from losses import GANLoss, task_loss
from utils import decode_segmap, dice_eval, read_lists, PSNR
from torch.utils.tensorboard import SummaryWriter
import medpy.metric.binary as mmb

import model

class SIFA_pytorch(nn.Module):
    def __init__(self, config):
        current_time = datetime.now().strftime("%Y%m%d-%H%M%S")
        # self._source_train_pth = config['source_train_pth']
        # self._target_train_pth = config['target_train_pth']
        # self._train_pth = config['_train_pth']
        # self._source_val_pth = config['source_val_pth']
        # self._target_val_pth = config['target_val_pth']
        # self._output_root_dir = config['output_root_dir']
        # self._images_dir = os.path.join(self._output_dir, 'imgs')
        # self._num_imgs_to_save = 20
        # self._pool_size = int(config['pool_size'])
        # self._lambda_a = float(config['_LAMBDA_A'])
        # self._lambda_b = float(config['_LAMBDA_B'])
        # self._skip = bool(config['skip'])
        # self._num_cls = int(config['num_cls'])
        # self._base_lr = float(config['base_lr'])
        # self._is_training_value = False
        self._batch_size = int(config['batch_size'])
        # self._lr_gan_decay = bool(config['lr_gan_decay'])
        # self._to_restore = bool(config['to_restore'])
        # self._checkpoint_dir = config['checkpoint_dir']
        # self.num_fake_inputs = 0


        self.direction = 'MR2CT'
        self.save_interval = 300
        self.print_freq = self.save_interval
        super(SIFA_pytorch, self).__init__()

        seg_beta = (0.5, 0.999)

        self.netG_A = model.ResnetGenerator()
        init_weights(self.netG_A.cuda(), init_gain=0.02)
        self.netEn = model.segmenter()
        init_weights(self.netEn.cuda(), init_gain=0.01)
        self.netDe = model.decoder()    
        init_weights(self.netDe.cuda(), init_gain=0.02) #### inconsistency: init_gain mismatch
        self.netClass = model.Build_classifier()
        init_weights(self.netClass.cuda(), init_gain=0.01)


        self.all_nets = [self.netG_A, self.netEn, self.netDe]
        self.model_names = ['G_A', 'En', 'De', 'Class']
        # self.path = '/home/xinwen/Desktop/Stage1-toggling/sifa-pytoch/output/20211127-170751/epoch_Best.ckpt'
        self.best_dice = 0.
        self.best_model = ''

        print("SIFA module loaded (test time)")

    def load_network(self, path):
        ckpt = torch.load(path)
        self.netG_A.load_state_dict(ckpt['G_A'])
        self.netEn.load_state_dict(ckpt['En'])
        self.netClass.load_state_dict(ckpt['Class'])

    def set_input(self, input):
        """Unpack input data from the dataloader and perform necessary pre-processing steps.
        Parameters:
            input (dict): include the data itself and its metadata information.
        The option 'direction' can be used to swap domain A and domain B.
        """
        AtoB = self.direction == 'CT2MR'
        self.real_A = input['A' if AtoB else 'B'].cuda()
        self.real_B = input['B' if AtoB else 'A'].cuda()
        self.image_paths = input['A_paths' if AtoB else 'B_paths']
        self.A_mask = input['A_mask' if AtoB else 'B_mask'].squeeze(1).cuda()
        self.B_mask = input['B_mask' if AtoB else 'A_mask'].squeeze(1).cuda()

    # def load_network(self, path):
    #     ckpt = torch.load(path)
    #     self.netG_A.load_state_dict(ckpt['G_A'])
    #     self.netEn.load_state_dict(ckpt['En'])
    #     self.netDe.load_state_dict(ckpt['De'])
    #     self.netClass.load_state_dict(ckpt['Class'])
    #     self.netClass_ll.load_state_dict(ckpt['Class_ll'])
    #     self.netD_A.load_state_dict(ckpt['D_A'])
    #     self.netD_B_aux.load_state_dict(ckpt['DC'])
    #     self.netD_B.load_state_dict(ckpt['D_B'])
    #     self.netD_P.load_state_dict(ckpt['D_P'])
    #     self.netD_P_ll.load_state_dict(ckpt['D_P_ll'])
    #     self.optimizer_DA.load_state_dict(ckpt['optimDA'])
    #     self.optimizer_DB.load_state_dict(ckpt['optimDB'])
    #     self.optimizer_DB_aux.load_state_dict(ckpt['optimDC'])
    #     self.optimizer_GA.load_state_dict(ckpt['optimGA'])
    #     self.optimizer_GB.load_state_dict(ckpt['optimGB'])
    #     self.optimizer_DP.load_state_dict(ckpt['optimDP'])
    #     self.optimizer_DP_ll.load_state_dict(ckpt['optimDP_ll'])
    #     self.optimizer_Seg.load_state_dict(ckpt['optimSeg'])

    def synth_fake_target(self, input):
        return self.netG_A(input).detach()
    
    def segmentate_test_set(self,input, gt):
        latent, _ = self.netEn(input)
        pred = self.netClass(latent)
        loss_Seg = task_loss(pred, gt).item()
        pred = torch.argmax(nn.Softmax(1)(pred), 1)
        return pred.squeeze(1), loss_Seg 

    def eval(self):
        for net in self.all_nets:
            net.eval()

    def eval_model(self):

        self.eval()
        # self.load_network(self.path)
        testset = abdom_dataset('./dataset/abdom', train=False)
        test_loader = torch.utils.data.DataLoader(
            testset,
            batch_size=self._batch_size,
            shuffle=True,
            num_workers=1,
            drop_last= False,
            pin_memory=True
        )
        all_pred =[]
        all_label=[]
        dice =[]
        assd=[]
        test_case_idx=0

        with torch.no_grad():
            # self.eval()
            for i, data in enumerate(test_loader):
                # print(i)

                self.set_input(data)
                # print(self.real_A.size())
                pred, test_seg_loss = self.segmentate_test_set(self.real_B, self.B_mask)
                all_pred.append(pred)
                all_label.append(self.B_mask)

            all_pred = torch.cat(all_pred, 0)
            all_label = torch.cat(all_label, 0)
            # print(all_pred.size())
            if self.direction == 'CT2MR':
                slices = [0, 36, 34+36, 39+34+36, 30+39+34+36] 
            else:
                slices = [0, 147, 147+149, 147+149+153, 147+149+153+144, 147+149+153+144+104, 147+149+153+144+104+90]

            for i in range(4):
                test_case_idx+=1
                case_pred = all_pred[slices[i]:slices[i+1]]
                case_label = all_label[slices[i]:slices[i+1]]
                # print(case_pred.size())
                case_label = case_label.argmax(1)

                for c in range(1, 5):
                    pred_test_data_tr = case_pred.clone().cpu().numpy()
                    pred_test_data_tr[pred_test_data_tr != c] = 0

                    
                    # print(all_label.size())
                    pred_gt_data_tr = case_label.clone().cpu().numpy()
                    pred_gt_data_tr[pred_gt_data_tr != c] = 0

                    # print(pred_test_data_tr.shape, pred_gt_data_tr.shape)

                    dice.append(mmb.dc(pred_test_data_tr, pred_gt_data_tr))
                    if np.sum(pred_test_data_tr) == 0:
                        pred_test_data_tr = pred_test_data_tr + 1e-7
                    assd.append(mmb.assd(pred_test_data_tr, pred_gt_data_tr))

                

            dice_avg = np.mean(dice)
            assd_avg = np.mean(assd)
            self.train()


            dice_arr = 100 * np.reshape(dice, [4, -1]).transpose()

            dice_mean = np.mean(dice_arr, axis=1)
            dice_std = np.std(dice_arr, axis=1)

            print( 'Dice:')
            print ('Liver :%.1f(%.1f)' % (dice_mean[0], dice_std[0]))
            print ('R Kidney:%.1f(%.1f)' % (dice_mean[1], dice_std[1]))
            print ('L Kidney:%.1f(%.1f)' % (dice_mean[2], dice_std[2]))
            print ('Spleen:%.1f(%.1f)' % (dice_mean[3], dice_std[3]))
            print ('Mean:%.1f' % np.mean(dice_mean))

            print('std: ', np.std(dice_mean))

            assd_arr = np.reshape(assd, [4, -1])

            assd_mean = np.mean(assd_arr, axis=1)
            assd_std = np.std(assd_arr, axis=1)

            print ('ASSD:')
            print ('Liver :%.1f(%.1f)' % (assd_mean[3], assd_std[3]))
            print ('R Kidney:%.1f(%.1f)' % (assd_mean[1], assd_std[1]))
            print ('L Kidney:%.1f(%.1f)' % (assd_mean[2], assd_std[2]))
            print ('Spleen:%.1f(%.1f)' % (assd_mean[0], assd_std[0]))
            print ('Mean:%.1f' % np.mean(assd_mean))

            if self.best_dice < dice_avg:
                self.best_dice = dice_mean
                self.best_model = self.path
                self.best_model_assd = assd_mean

            with open('abdom_accuracies', 'a') as fp:
                fp.writelines(self.path+ '\n')
                fp.writelines('mask dice: {:3f}\n'.format(dice_avg))
                fp.writelines('assd dice: {:3f}\n'.format(assd_avg))

            # row1 = [(visual_data2.repeat(3,1,1)+1)/2, visual_pred2, visual_gt2]
            # row1 = torch.cat(row1, 2)
            # row2 = [(visual_data1.repeat(3,1,1)+1)/2, visual_pred1, visual_gt1]
            # row2 = torch.cat(row2, 2)

            # pic = torch.cat((row1,row2), 1)  # C=3, H*2, W*4




if __name__ == '__main__':
    manual_seed=1234
    random.seed(manual_seed)
    np.random.seed(manual_seed)
    torch.manual_seed(manual_seed)
    torch.cuda.manual_seed_all(manual_seed)

    with open('config_param.json') as config_file:
        config = json.load(config_file)

    folder_name = '20220228-120035'

    with open('./output/'+folder_name+'/checkpoint_names', 'r') as fd:
        list = fd.readlines()

    checkpoint_names = []

    for item in list:
        checkpoint_names.append(item[:-1])
        print(item[:-1])

    sifa_model = SIFA_pytorch(config)
    for checkpoint in checkpoint_names:
        sifa_model.path = './output/'+ folder_name + '/'+checkpoint
        # sifa_model.path = './output/20211120-011711/iter_13199.ckpt'
        # print(sifa_model.checkpoint_pth)
        sifa_model.eval_model()

    with open('Best_models_in_folders', 'a') as fp:
        fp.writelines('Best model in folder: '+sifa_model.best_model+ '\n')
        fp.writelines('mask dice: {:3f}\n'.format(np.mean(sifa_model.best_dice)))
        fp.writelines('assd dice: {:3f}\n'.format(np.mean(sifa_model.best_model_assd)))
        
        fp.writelines( 'Dice:')
        fp.writelines('Liver :%.1f(%.1f)' % (sifa_model.best_dice[0]))
        fp.writelines('R Kidney:%.1f(%.1f)' % (sifa_model.best_dice[1]))
        fp.writelines('L Kidney:%.1f(%.1f)' % (sifa_model.best_dice[2]))
        fp.writelines('Splen:%.1f(%.1f)' % (sifa_model.best_dice[3]))
        fp.writelines('Mean:%.1f' % np.mean(dice_mean))

        fp.writelines('ASSD:')
        fp.writelines('Liver :%.1f(%.1f)' % (sifa_model.best_model_assd[0]))
        fp.writelines('R Kidney:%.1f(%.1f)' % (sifa_model.best_model_assd[1]))
        fp.writelines('L Kidney:%.1f(%.1f)' % (sifa_model.best_model_assd[2]))
        fp.writelines('Spleen:%.1f(%.1f)' % (sifa_model.best_model_assd[3]))
        fp.writelines('Mean:%.1f' % np.mean(assd_mean))


    # sifa_model = SIFA_pytorch(config)
    # sifa_model.path = '/home/xinwen/Desktop/Stage1-toggling/gitmixup/output/65.5/65.5.ckpt'
    # # # print(sifa_model.checkpoint_pth)
    # sifa_model.eval_model()