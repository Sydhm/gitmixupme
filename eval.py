from datetime import datetime
import json
import numpy as np
import random
import os
import cv2
import time
from basic_blocks import init_weights
from dataloader import UnalignedDataset
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

        self.load_network(self.path)

        ################## test files direction keys
        source = ['ct', 'CT']
        target = ['mr', 'MR']
        ################## test files
        source_txt_path = '/home/xinwen/Downloads/SIFA-master/data/datalist/test_' +source[0]+ '.txt'
        source_test_list = read_lists(source_txt_path)
        target_txt_path = '/home/xinwen/Downloads/SIFA-master/data/datalist/test_' +target[0]+ '.txt'
        target_test_list = read_lists(target_txt_path)
        data_size = [1, 256, 256]
        label_size = [1, 256, 256]
        # batch_size = 2

        if "ct" in source_txt_path:
            src_Max = 3.2
            src_Min = -2.8
        else:  #### mr
            src_Max = 4.4
            src_Min = -1.8

        if "mr" in target_txt_path:
            tgt_Max = 4.4
            tgt_Min = -1.8
            tgt_display_idx = np.array([[2, 50],
                                        [1, 83]])
        else:  #### ct
            tgt_Max = 3.2
            tgt_Min = -2.8
            tgt_display_idx = np.array([[3, 140],
                                        [1, 77]])

            ### eval
        self.eval()
        with torch.no_grad():
            print("EVAL MODE")
            test_case_idx = -1
            psnr = []

            # EVAL SYSN
            # https://github.com/photosynthesis-team/piq/issues/241
            for idx_file, fid in enumerate(source_test_list):
                test_case_idx += 1
                # print("FILE ", test_case_idx)
                _npz_dict = np.load(fid)
                data = (_npz_dict['arr_0']-src_Min)/(src_Max-src_Min)*2-1   # 256, 256,  256 H W frame
                label = _npz_dict['arr_1']

                if True:
                    data = np.flip(data, axis=0)
                    data = np.flip(data, axis=1)
                    label = np.flip(label, axis=0)
                    label = np.flip(label, axis=1)

                tmp_pred = np.zeros(label.shape)

                frame_list = [kk for kk in range(data.shape[2])]
                for ii in range(int(np.floor(data.shape[2] // self._batch_size))):
                    # print(ii)
                    data_batch = torch.zeros(self._batch_size, data_size[0], data_size[1], data_size[2])
                    for idx, jj in enumerate(frame_list[ii * self._batch_size: (ii + 1) * self._batch_size]): 
                        data_batch[idx, ...] = torch.from_numpy(data[..., jj].copy()).unsqueeze(0)
                    input= data_batch.cuda()
                    fake_target= self.synth_fake_target(input)

                    psnr.append(PSNR()(input, fake_target).item())

            ###### Seg eval
            dice = np.zeros([4, 4])  # 4 test sets, 4 classes
            assd = np.zeros([4, 4])
            test_seg_losses = []
            test_case_idx = -1

            for idx_file, fid in enumerate(target_test_list):
                test_case_idx += 1
                # print("FILE ", test_case_idx)
                _npz_dict = np.load(fid)
                data = (_npz_dict['arr_0']-tgt_Min)/(tgt_Max-tgt_Min)*2-1   # 256, 256,  256 H W frame
                label = _npz_dict['arr_1']

                if True:
                    data = np.flip(data, axis=0)
                    data = np.flip(data, axis=1)
                    label = np.flip(label, axis=0)
                    label = np.flip(label, axis=1)

                tmp_pred = torch.zeros(label.shape, device=torch.device('cuda')) ###np.zeros(label.shape)

                frame_list = [kk for kk in range(data.shape[2])]
                for ii in range(int(np.floor(data.shape[2] // self._batch_size))):
                    # print(ii)
                    data_batch = torch.zeros((self._batch_size, data_size[0], data_size[1], data_size[2]), device=torch.device('cuda'))
                    label_batch = torch.zeros((self._batch_size, label_size[1], label_size[2]), device=torch.device('cuda'))
                    for idx, jj in enumerate(frame_list[ii * self._batch_size: (ii + 1) *self._batch_size]):
                        data_batch[idx, ...] = torch.from_numpy(data[..., jj].copy()).unsqueeze(0)
                        label_batch[idx, ...] = torch.from_numpy(label[..., jj].copy())
                    label_batch =  torch.nn.functional.one_hot(label_batch.long(), 5)
                    label_batch = (label_batch.float()).permute(0,3,1,2)
                    #####
                    input, gt = data_batch.cuda(), label_batch.cuda()
                    pred, test_seg_loss = self.segmentate_test_set(input, gt)
                    test_seg_losses.append(test_seg_loss)

                    for idx, jj in enumerate(frame_list[ii * self._batch_size: (ii + 1) * self._batch_size]):
                        tmp_pred[..., jj] = pred[idx, ...]

                if test_case_idx == tgt_display_idx[0,0]:   ### CT 1003  -- https://github.com/cchen-cc/SIFA/issues/31
                    visual_data1 = torch.from_numpy(data[..., tgt_display_idx[0,1]].copy()).unsqueeze(0).float().cuda()
                    visual_pred1 = decode_segmap((tmp_pred[..., tgt_display_idx[0,1]]))
                    visual_gt1 = decode_segmap(torch.from_numpy(label[..., tgt_display_idx[0,1]].copy()))

                if test_case_idx == tgt_display_idx[1,0]:   ### CT 1003  -- https://github.com/cchen-cc/SIFA/issues/31
                    visual_data2 = torch.from_numpy(data[..., tgt_display_idx[1,1]].copy()).unsqueeze(0).float().cuda()
                    visual_pred2 = decode_segmap((tmp_pred[..., tgt_display_idx[1,1]]))
                    visual_gt2 = decode_segmap(torch.from_numpy(label[..., tgt_display_idx[1,1]].copy()))

                tmp_pred = tmp_pred.detach().cpu().numpy()
                for c in range(1, 5):
                    pred_test_data_tr = tmp_pred.copy()
                    pred_test_data_tr[pred_test_data_tr != c] = 0

                    pred_gt_data_tr = label.copy()
                    pred_gt_data_tr[pred_gt_data_tr != c] = 0

                    dice[test_case_idx, c-1]=mmb.dc(pred_test_data_tr, pred_gt_data_tr)
                    if np.sum(pred_test_data_tr) == 0:
                        pred_test_data_tr = pred_test_data_tr + 1e-7
                    assd[test_case_idx, c-1]=mmb.assd(pred_test_data_tr, pred_gt_data_tr)

            print(dice)
            print(assd)
            dice_avg = np.mean(dice)
            assd_avg = np.mean(assd)

            if self.best_dice < dice_avg:
                self.best_dice = dice_avg
                self.best_model = self.path
                self.best_model_assd = assd_avg

            with open('target_seg_accuracy_20211126-125532', 'a') as fp:
                fp.writelines(self.path+ '\n')
                fp.writelines('mask dice: {:3f}\n'.format(dice_avg))
                fp.writelines('assd dice: {:3f}\n'.format(assd_avg))

            row1 = [(visual_data2.repeat(3,1,1)+1)/2, visual_pred2, visual_gt2]
            row1 = torch.cat(row1, 2)
            row2 = [(visual_data1.repeat(3,1,1)+1)/2, visual_pred1, visual_gt1]
            row2 = torch.cat(row2, 2)

            pic = torch.cat((row1,row2), 1)  # C=3, H*2, W*4




if __name__ == '__main__':
    manual_seed=1234
    random.seed(manual_seed)
    np.random.seed(manual_seed)
    torch.manual_seed(manual_seed)
    torch.cuda.manual_seed_all(manual_seed)

    with open('config_param.json') as config_file:
        config = json.load(config_file)

    folder_name = 'patch_ad'

    with open('./output/'+folder_name+'/checkpoint_names', 'r') as fd:
        list = fd.readlines()

    checkpoint_names = []

    for item in list:
        checkpoint_names.append(item[:-1])
        print(item[:-1])

    sifa_model = SIFA_pytorch(config)
    # sifa_model.path = './output/20211120-011711/iter_13199.ckpt'
    # # print(sifa_model.checkpoint_pth)
    # sifa_model.eval_model()

    for checkpoint in checkpoint_names:
        sifa_model.path = './output/'+ folder_name + '/'+checkpoint
        # sifa_model.path = './output/20211120-011711/iter_13199.ckpt'
        # print(sifa_model.checkpoint_pth)
        sifa_model.eval_model()

    with open('Best_models_in_folders', 'a') as fp:
        fp.writelines('Best model in folder: '+sifa_model.best_model+ '\n')
        fp.writelines('mask dice: {:3f}\n'.format(sifa_model.best_dice))
        fp.writelines('assd dice: {:3f}\n'.format(sifa_model.best_model_assd))
    # for item in list:
    #     checkpoint_names.append(item[5:-1])
    #     print(item[5:-1])

    # sifa_model = SIFA_pytorch(config)

    # for checkpoint in checkpoint_names:
    #     sifa_model.path = './output/'+ folder_name + '/iter'+checkpoint
    #     # print(sifa_model.checkpoint_pth)
    #     sifa_model.eval_model()

