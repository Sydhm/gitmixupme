# This training version is to have DB_AUX take in real target as real and source+fake target as fake (selective update) during discriminator optimization

from datetime import datetime
import json
from pickle import bytes_types
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
from transformer import ViT ## 3 convs lead to 8*8 patchembedding
from ad_net import Domain_classifier, DC, NLayerDc

# torch.autograd.set_detect_anomaly(True)

class SIFA_pytorch(nn.Module):
    def __init__(self, config):
        current_time = datetime.now().strftime("%Y%m%d-%H%M%S")
        # self._source_train_pth = config['source_train_pth']
        # self._target_train_pth = config['target_train_pth']
        self._train_pth = config['_train_pth']
        self._output_root_dir = config['output_root_dir']
        if not os.path.isdir(self._output_root_dir):
            os.makedirs(self._output_root_dir)
        self._output_dir = os.path.join(self._output_root_dir, current_time)
        self._images_dir = os.path.join(self._output_dir, 'imgs')
        self._num_imgs_to_save = 20
        self._pool_size = int(config['pool_size'])
        self._lambda_a = float(config['_LAMBDA_A'])
        self._lambda_b = float(config['_LAMBDA_B'])
        self._skip = bool(config['skip'])
        self._num_cls = int(config['num_cls'])
        self._base_lr = float(config['base_lr'])
        self._is_training_value = bool(config['is_training_value'])
        self._batch_size = int(config['batch_size'])
        self._lr_gan_decay = bool(config['lr_gan_decay'])
        self._to_restore = bool(config['to_restore'])
        self._checkpoint_dir = config['checkpoint_dir']
        self.num_fake_inputs = 0

        self.fake_images_A = torch.zeros(
            (self._pool_size, self._batch_size, 1,  model.IMG_HEIGHT, model.IMG_WIDTH), device=torch.device('cuda'))
        self.fake_images_B = torch.zeros(
            (self._pool_size, self._batch_size, 1,  model.IMG_HEIGHT, model.IMG_WIDTH), device=torch.device('cuda'))


        self.direction = 'MR2CT'
        self.save_interval = 300
        self.print_freq = self.save_interval
        super(SIFA_pytorch, self).__init__()

        gan_beta = (0.5, 0.999)

        self.netG_A = model.ResnetGenerator()
        init_weights(self.netG_A.cuda(), init_gain=0.02)
        self.netEn = model.segmenter()
        init_weights(self.netEn.cuda(), init_gain=0.01)
        self.netDe = model.decoder()    
        init_weights(self.netDe.cuda(), init_gain=0.02) #### inconsistency: init_gain mismatch
        self.netClass = model.Build_classifier()
        init_weights(self.netClass.cuda(), init_gain=0.01)
        self.netClass_ll = model.Build_classifier()
        init_weights(self.netClass_ll.cuda(), init_gain=0.01)
        self.netD_A = model.NLayerDiscriminator_aux()
        init_weights(self.netD_A.cuda(), init_gain=0.02)
        self.netD_B = model.NLayerDiscriminator()
        init_weights(self.netD_B.cuda(), init_gain=0.02)
        self.netD_B_aux = NLayerDc(1)
        init_weights(self.netD_B_aux.cuda(), init_gain=0.01)
        self.netD_P = model.NLayerDiscriminator(input_nc=5)   #### channels = 5
        init_weights(self.netD_P.cuda(), init_gain=0.02)
        self.netD_P_ll = model.NLayerDiscriminator(input_nc=5)    ##### channels = 5
        init_weights(self.netD_P_ll.cuda(), init_gain=0.02)

        self.all_nets = [self.netG_A, self.netEn, self.netDe, self.netClass, self.netClass_ll, self.netD_A, self.netD_B, self.netD_P, self.netD_P_ll]
        self.model_names = ['G_A', 'En', 'De', 'Class', 'Class_ll', 'D_A', 'D_B', 'D_P', 'D_P_ll']

        # TOTAL_PARAMS = 0
        # for name, net in zip(self.model_names, self.all_nets):
        #     # if name == 'D_P':
        #         # print(name)
        #         TOTAL_PARAMS += sum(p.numel() for p in net.parameters() if p.requires_grad)
        #         # print("Trainable params: ", sum(p.numel() for p in net.parameters() if p.requires_grad))
        #         print(name, sum(p.numel() for p in net.parameters() if p.requires_grad))
        #         # for name, param in net.named_parameters():
        #         #     if param.requires_grad:
        #         #         print(name, param.numel())
        # print("Trainable params: ", TOTAL_PARAMS)

        # quit()

        self.criterionCycle = torch.nn.L1Loss().cuda()
        self.criterionGAN = GANLoss().cuda()
        self.task_loss = task_loss

        self.optimizer_DA = torch.optim.Adam(self.netD_A.parameters(), lr = self._base_lr, betas=gan_beta)
        self.optimizer_DB = torch.optim.Adam(self.netD_B.parameters(), lr = self._base_lr, betas=gan_beta)
        # DB_aux_params = list(self.netD_B_aux.parameters())
        # print(len(self.netD_B_aux.ad_list))
        # for ad_num in range(len(self.netD_B_aux.ad_list)):
        #     DB_aux_params += list(self.netD_B_aux.ad_list[ad_num].parameters())
        # self.optimizer_DB_aux = torch.optim.Adam(DB_aux_params, lr = self._base_lr, betas=gan_beta)
        self.optimizer_DB_aux = torch.optim.Adam(self.netD_B_aux.parameters(), lr = self._base_lr, betas=gan_beta)
        self.optimizer_GA = torch.optim.Adam(self.netG_A.parameters(), lr = self._base_lr, betas=gan_beta)
        self.optimizer_GB = torch.optim.Adam(self.netDe.parameters(), lr = self._base_lr, betas=gan_beta)
        self.optimizer_DP = torch.optim.Adam(self.netD_P.parameters(), lr = self._base_lr, betas=gan_beta)
        self.optimizer_DP_ll = torch.optim.Adam(self.netD_P_ll.parameters(), lr = self._base_lr, betas=gan_beta)
        self.optimizer_Seg = torch.optim.Adam(itertools.chain(self.netEn.parameters(), self.netClass.parameters(), self.netClass_ll.parameters()), lr = self._base_lr, weight_decay=0.0001)
        
        all_optim = [self.optimizer_DA, self.optimizer_DB, self.optimizer_DB_aux, self.optimizer_GA, self.optimizer_GB, self.optimizer_DP, self.optimizer_DP_ll, self.optimizer_Seg]
        def lambda_rule(epoch):
            lr_l = 1.0 - max(0, epoch - 16) / float(70 + 1)
            return lr_l
        # self.scheduler = torch.optim.lr_scheduler.LambdaLR(all_optim, lr_lambda=lambda_rule)
        self.schedulers = [torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lambda_rule) for optimizer in all_optim]

        print("SIFA module initiated")


#################################### member functions ################################
    def set_requires_grad(self, nets, requires_grad=False):
        """Set requies_grad=Fasle for all the networks to avoid unnecessary computations
        Parameters:
            nets (network list)   -- a list of networks
            requires_grad (bool)  -- whether the networks require gradients or not
        """
        if not isinstance(nets, list):
            nets = [nets]
        for net in nets:
            if net is not None:
                for param in net.parameters():
                    param.requires_grad = requires_grad

    # def zero_grad(self):
    #     for net in self.all_nets:
    #         net.zero_grad()
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

        # # # random pixel
        # self.bit_mask = torch.cuda.FloatTensor(self.real_A.size()).uniform_() > 0.55
        # self.bit_mask = self.bit_mask.float()

        # random 8*8 patch
        mix_ratio = np.random.beta(2.0, 2.0)
        self.bit_mask = torch.cuda.FloatTensor(self._batch_size,1,8,8).uniform_() > mix_ratio #0.5
        self.bit_mask88 = self.bit_mask.float()
        if torch.mean(self.bit_mask88)==1 or torch.mean(self.bit_mask88)==0:
            while torch.mean(self.bit_mask88)==1 or torch.mean(self.bit_mask88)==0:
                self.bit_mask = torch.cuda.FloatTensor(self._batch_size,1,8,8).uniform_() > 0.55
                self.bit_mask88 = self.bit_mask.float()
        # self.bit_mask88 = self.bit_mask.float()
        self.bit_mask = torch.nn.functional.interpolate(self.bit_mask88, [model.IMG_HEIGHT, model.IMG_WIDTH], mode='area')
        # self.bit_mask88 = torch.nn.functional.interpolate(self.bit_mask88, [24, 24], mode='area')

    def cyclegan_one_pass(self):
        self.fake_B = self.netG_A(self.real_A) # skip done in forward
        self.latent_fake_B, self.latent_fake_B_ll = self.netEn(self.fake_B)
        self.cycle_A = self.netDe(self.latent_fake_B, self.fake_B)

        self.latent_B, self.latent_B_ll = self.netEn(self.real_B)
        self.fake_A = self.netDe(self.latent_B, self.real_B)
        self.cycle_B = self.netG_A(self.fake_A)

    def fake_image_pool(self, num_fakes, fake, fake_pool):
        if num_fakes < self._pool_size:
            fake_pool[num_fakes] = fake
            return fake
        else:
            p = random.random()
            if p > 0.5:
                random_id = random.randint(0, self._pool_size - 1)
                temp = fake_pool[random_id]
                fake_pool[random_id] = fake
                return temp
            else:
                return fake
################################### optimization functions ##############################
    def compute_d_a_loss(self):
        self.cyclegan_one_pass()
        pred_fake, pred_fake_aux = self.netD_A(self.fake_A_temp)
        fake_loss = self.criterionGAN(pred_fake, False).mean() + self.criterionGAN(pred_fake_aux, False).mean()

        pred_real, _ = self.netD_A(self.real_A)
        _, pred_cycle_A = self.netD_A(self.cycle_A.detach())
        real_loss = self.criterionGAN(pred_real, True).mean() + self.criterionGAN(pred_cycle_A, True).mean()
        DA_loss = (fake_loss + real_loss)*0.5
        return DA_loss
    def compute_g_a_loss(self):
        self.cyclegan_one_pass()
        cycle_loss_a = self._lambda_a * self.criterionCycle(self.real_A, self.cycle_A)
        cycle_loss_b = self._lambda_b * self.criterionCycle(self.real_B, self.cycle_B)

        DB_pred =  self.netD_B(self.fake_B) 
        g_a_ganloss = self.criterionGAN(DB_pred, True).mean()

        pred_mixed_m, m = self.netD_B_aux(self.real_A*(1-self.bit_mask)+self.fake_B*self.bit_mask)  ### (1-m)real A + (m)fake B
        self.bit_mask_pred = pred_mixed_m.detach()
        m_gt = torch.mean(self.bit_mask88.reshape(self._batch_size,-1), dim=1, keepdim=True)   # 8, 1
        # print(pred_mixed_m.size())
        local_loss = self.criterionGAN(pred_mixed_m, self.bit_mask88)
        g_a_ganloss_m = local_loss.mean() + self.criterionGAN(m, m_gt).mean()
        # print(g_a_ganloss_m.size())
        pred_mixed_notm, notm = self.netD_B_aux(self.real_A*self.bit_mask + self.fake_B*(1 -self.bit_mask))  ### (m)real A + (1-m)fake B
        notm_gt = torch.mean((1-self.bit_mask88).reshape(self._batch_size,-1), dim=1, keepdim=True)   # 8, 1
        local_loss = self.criterionGAN(pred_mixed_notm, 1-self.bit_mask88)
        g_a_ganloss_notm = (local_loss*(1 -self.bit_mask88)).sum()/(1 -self.bit_mask88).sum() + self.criterionGAN(notm, notm_gt).mean()
        # print(self.criterionGAN(notm, 1-self.bit_mask88.sum()/64).size())
        g_a_ganloss += (g_a_ganloss_m + g_a_ganloss_notm)*0.1


        ga_loss = cycle_loss_a + cycle_loss_b + g_a_ganloss
        self.fake_B_temp = self.fake_image_pool(self.num_fake_inputs, self.fake_B.detach(), self.fake_images_B)

        return ga_loss

    def compute_g_b_loss(self):
        self.cyclegan_one_pass()
        cycle_loss_a = self._lambda_a * self.criterionCycle(self.real_A, self.cycle_A)
        cycle_loss_b = self._lambda_b * self.criterionCycle(self.real_B, self.cycle_B)

        DA_pred, _ =  self.netD_A(self.fake_A) 
        g_b_ganloss = self.criterionGAN(DA_pred, True).mean()

        pred_mixed_m, m = self.netD_B_aux(self.real_B*self.bit_mask+self.fake_A*(1-self.bit_mask))  ### (1-m)real A + (m)fake B
        m_gt = torch.mean(self.bit_mask88.reshape(self._batch_size,-1), dim=1, keepdim=True)   # 8, 1
        # print(m_gt.size(), 'm gt size')
        local_loss = self.criterionGAN(pred_mixed_m, self.bit_mask88)
        g_b_ganloss_m = (local_loss*(1-self.bit_mask88)).sum()/(1-self.bit_mask88).sum()+self.criterionGAN(m, m_gt).mean()
        ###### PSEUDO ------------------      A: 0,     fake A: assgined 0*     (weight)1
        pred_mixed_notm, notm = self.netD_B_aux(self.real_B*(1-self.bit_mask) + self.fake_A*self.bit_mask)  ### (m)real A + (1-m)fake B
        notm_gt = torch.mean((1-self.bit_mask88).reshape(self._batch_size,-1), dim=1, keepdim=True)   # 8, 1
        local_loss = self.criterionGAN(pred_mixed_notm, 1-self.bit_mask88)
        g_b_ganloss_notm = (local_loss*self.bit_mask88).sum()/self.bit_mask88.sum() +self.criterionGAN(notm, notm_gt).mean()
        g_b_ganloss += (g_b_ganloss_m + g_b_ganloss_notm)*0.1


        gb_loss = cycle_loss_a + cycle_loss_b + g_b_ganloss
        self.fake_A_temp = self.fake_image_pool(self.num_fake_inputs, self.fake_A.detach(), self.fake_images_A)
        return gb_loss

    def compute_d_b_loss(self):
        fake_B = self.fake_B_temp # on gpu as image stored is cuda.tensor, detached when pooled

        pred_fake = self.netD_B(fake_B)
        fake_loss = self.criterionGAN(pred_fake, False).mean()

        pred_real = self.netD_B(self.real_B)
        real_loss = self.criterionGAN(pred_real, True).mean()

        DB_loss = (fake_loss + real_loss)*0.5
        return DB_loss

    def compute_d_b_aux_loss(self): # source -> 0, target -> 1 -------------------------- A:0, B:1
        fake_B = self.fake_B_temp # on gpu as image stored is cuda.tensor, detached when pooled

        amix = self.real_A*(1-self.bit_mask) + fake_B*self.bit_mask
        pred_fakes, f = self.netD_B_aux(amix.detach())
        fake_loss = self.criterionGAN(pred_fakes, False).mean() + self.criterionGAN(f, False).mean()


        bmix = self.real_B*self.bit_mask + (self.fake_A.detach())*(1-self.bit_mask)   ## double adversarial
        pred_reals, t = self.netD_B_aux(bmix.detach())
        real_loss = self.criterionGAN(pred_reals, True).mean()+self.criterionGAN(t, True).mean()

        pred_real, t = self.netD_B_aux(self.real_B)  # B is 1
        # pred_real_reject = self.netD_B_aux(self.fake_A)
        real_loss = self.criterionGAN(pred_real, True).mean()+self.criterionGAN(t, True).mean()

        pred_false, f = self.netD_B_aux(self.real_A)  # A is 0
        # pred_false_reject = self.netD_B_aux(self.fake_B)
        fake_loss = self.criterionGAN(pred_false, False).mean() + self.criterionGAN(f, False).mean()

        pred_m, m = self.netD_B_aux(self.real_B*self.bit_mask+(1-self.bit_mask)*self.real_A)
        m_gt = torch.mean(self.bit_mask88.reshape(self._batch_size,-1), dim=1, keepdim=True)
        m_loss = self.criterionGAN(pred_m, self.bit_mask88).mean() + self.criterionGAN(m, m_gt).mean()

        # pred_notm = self.netD_B_aux(self.real_A*self.bit_mask+(1-self.bit_mask)*self.real_B)
        # # m_gt = torch.mean(self.bit_mask88.reshape(self._batch_size,-1), dim=1, keepdim=True)
        # m_loss = self.criterionGAN(pred_notm, 1-self.bit_mask88).mean() #+ self.criterionGAN(m, m_gt).mean()

        DB_aux_loss = (real_loss + fake_loss + m_loss)*0.5
        return DB_aux_loss

    def compute_d_p_loss(self):
        fake_B = self.netG_A(self.real_A) # skip done in forward
        latent_fake_B, _ = self.netEn(fake_B)
        latent_B, _ = self.netEn(self.real_B)

        pred_A = self.netClass(latent_fake_B).detach()
        pred_B = self.netClass(latent_B).detach()
        pred_fake = self.netD_P(pred_B)
        fake_loss = self.criterionGAN(pred_fake, False).mean()
        pred_real = self.netD_P(pred_A)
        real_loss = self.criterionGAN(pred_real, True).mean()

        DP_loss = (fake_loss + real_loss)*0.5

        return DP_loss

    def compute_d_p_ll_loss(self):
        fake_B = self.netG_A(self.real_A) # skip done in forward
        _, latent_fake_B_ll = self.netEn(fake_B)
        _, latent_B_ll = self.netEn(self.real_B)

        pred_A_ll = self.netClass_ll(latent_fake_B_ll).detach()
        pred_B_ll = self.netClass_ll(latent_B_ll).detach()
        pred_fake = self.netD_P_ll(pred_B_ll)
        fake_loss = self.criterionGAN(pred_fake, False).mean()
        pred_real = self.netD_P_ll(pred_A_ll)
        real_loss = self.criterionGAN(pred_real, True).mean()

        DP_ll_loss = (fake_loss + real_loss)*0.5

        return DP_ll_loss

    def compute_seg_loss(self):
        self.cyclegan_one_pass()
        pred_A = self.netClass(self.latent_fake_B)
        pred_A_ll = self.netClass_ll(self.latent_fake_B_ll)
        task_loss = self.task_loss(pred_A, self.A_mask)
        task_loss_ll = 0.1*self.task_loss(pred_A_ll, self.A_mask)
        
        cycle_loss_a = self._lambda_a * self.criterionCycle(self.real_A, self.cycle_A)
        cycle_loss_b = self._lambda_b * self.criterionCycle(self.real_B, self.cycle_B)
        DA_pred, DA_pred_aux =  self.netD_A(self.fake_A) 
        g_b_ganloss = self.criterionGAN(DA_pred, True).mean()

        # pred_mixed_m = self.netD_B_aux(self.real_B*self.bit_mask+self.fake_A*(1-self.bit_mask))  ### (1-m)real A + (m)fake B
    
        # local_loss = self.criterionGAN(pred_mixed_m, self.bit_mask88)
        # g_b_ganloss_m = (local_loss*(1-self.bit_mask88)).sum()/(1-self.bit_mask88).sum()#+self.criterionGAN(m, m_gt).mean()
        # pred_mixed_notm = self.netD_B_aux(self.real_B*(1-self.bit_mask) + self.fake_A*self.bit_mask)  ### (m)real A + (1-m)fake B
        # local_loss = self.criterionGAN(pred_mixed_notm, 1-self.bit_mask88)
        # g_b_ganloss_notm = (local_loss*self.bit_mask88).sum()/self.bit_mask88.sum() #+self.criterionGAN(notm, notm_gt).mean()
        # g_b_ganloss += (g_b_ganloss_m + g_b_ganloss_notm)*0.1

        gb_loss = 0.1*(cycle_loss_a + cycle_loss_b + g_b_ganloss)

        g_b_ganloss_aux = 0.1*self.criterionGAN(DA_pred_aux, True).mean()    #### termed "lsgan_loss_a_aux" in sifa-tf,  fed fake_A
        
        pred_B = self.netClass(self.latent_B)
        pred_B_ll = self.netClass_ll(self.latent_B_ll)
        DP_pred = self.netD_P(pred_B)
        DP_pred_ll = self.netD_P_ll(pred_B_ll)
        DPi_losses = 0.1*self.criterionGAN(DP_pred, True).mean() + 0.01*self.criterionGAN(DP_pred_ll, True).mean()

        # time.sleep(5)
        return task_loss + task_loss_ll + gb_loss + g_b_ganloss_aux + DPi_losses

    def optimize_params(self):
        # Optimize G_A
        self.set_requires_grad(self.all_nets, False)
        self.set_requires_grad(self.netG_A, True)
        self.optimizer_GA.zero_grad()
        # self.zero_grad()
        loss_g_a = self.compute_g_a_loss()
        loss_g_a.backward()
        self.optimizer_GA.step()
        self.losses_ga=(loss_g_a.item())
        del loss_g_a

        # Optimize D_B
        self.set_requires_grad(self.all_nets, False)
        self.set_requires_grad(self.netD_B, True)
        self.optimizer_DB.zero_grad()
        # self.zero_grad()
        loss_d_b = self.compute_d_b_loss()
        loss_d_b.backward()
        self.optimizer_DB.step()
        self.losses_db=(loss_d_b.item())
        del loss_d_b

        # Optimize seg_B
        self.set_requires_grad(self.all_nets, False)
        self.set_requires_grad([self.netEn, self.netClass, self.netClass_ll], True)
        self.optimizer_Seg.zero_grad()  #### self.netEn.parameters(), self.netClass.parameters(), self.netClass_ll.parameters()
        # self.zero_grad()
        loss_seg = self.compute_seg_loss()
        loss_seg.backward()
        self.optimizer_Seg.step()
        self.losses_seg=(loss_seg.item())
        del loss_seg

        # Optimize G_B
        self.set_requires_grad(self.all_nets, False)
        self.set_requires_grad(self.netDe, True)
        self.optimizer_GB.zero_grad()
        loss_g_b = self.compute_g_b_loss()
        loss_g_b.backward()
        self.optimizer_GB.step()
        self.losses_gb=(loss_g_b.item())
        del loss_g_b

        # Optimize D_A
        self.set_requires_grad(self.all_nets, False)
        self.set_requires_grad(self.netD_A, True)
        self.optimizer_DA.zero_grad()
        # self.zero_grad()
        loss_d_a = self.compute_d_a_loss()
        loss_d_a.backward()
        self.optimizer_DA.step()
        self.losses_da=(loss_d_a.item())
        del loss_d_a

        # Optimize D_B_aux
        self.set_requires_grad(self.all_nets, False)
        self.set_requires_grad(self.netD_B_aux, True)
        self.optimizer_DB_aux.zero_grad()
        # self.zero_grad()
        loss_d_b_aux = self.compute_d_b_aux_loss()*0.1
        loss_d_b_aux.backward()
        self.optimizer_DB_aux.step()
        self.losses_db_aux=(loss_d_b_aux.item())
        del loss_d_b_aux

        # Optimize D_P
        self.set_requires_grad(self.all_nets, False)
        self.set_requires_grad(self.netD_P, True)
        self.optimizer_DP.zero_grad()
        # self.zero_grad()
        loss_d_p = self.compute_d_p_loss()
        loss_d_p.backward()
        self.optimizer_DP.step()

        # Optimize D_P_ll
        self.set_requires_grad(self.all_nets, False)
        self.set_requires_grad(self.netD_P_ll, True)
        self.optimizer_DP_ll.zero_grad()
        # self.zero_grad()
        loss_d_p_ll = self.compute_d_p_ll_loss()
        loss_d_p_ll.backward()
        self.optimizer_DP_ll.step()
        self.losses_dpi=(loss_d_p.item()+loss_d_p_ll.item())
        del loss_d_p
        del loss_d_p_ll

    def call_last_batch(self):
        with torch.no_grad():
            self.fake_B = self.netG_A(self.real_A)
            latent_fake_B, _ = self.netEn(self.fake_B)
            self.pred_fakeB = self.netClass(latent_fake_B)
            self.dice_fake_b=torch.mean(dice_eval(self.pred_fakeB, self.A_mask, self._num_cls)).item()
            latent_B, _ = self.netEn(self.real_B)
            self.pred_B = self.netClass(latent_B)
            self.dice_b=torch.mean(dice_eval(self.pred_B, self.B_mask, self._num_cls)).item()
########################### eval functions #############################
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

    def train(self):
        for net in self.all_nets:
            net.train()
########################### save&load model functions ##########################
    def save_networks(self, epoch):
        """Save all the networks to the disk.
        Parameters:
            epoch (int) -- current epoch; used in the file name '%s_net_%s.pth' % (epoch, name)
        """
        torch.save({'epoch': epoch,
                    'G_A': self.netG_A.state_dict(),
                    'En': self.netEn.state_dict(),
                    'De': self.netDe.state_dict(),
                    'Class': self.netClass.state_dict(),
                    'Class_ll': self.netClass_ll.state_dict(),
                    'D_A': self.netD_A.state_dict(),
                    'DC': self.netD_B_aux.state_dict(),
                    'D_B': self.netD_B.state_dict(),
                    'D_P': self.netD_P.state_dict(),
                    'D_P_ll': self.netD_P_ll.state_dict(),
                    'optimDA': self.optimizer_DA.state_dict(),
                    'optimDB': self.optimizer_DB.state_dict(),
                    'optimDC': self.optimizer_DB_aux.state_dict(),
                    'optimGA': self.optimizer_GA.state_dict(),
                    'optimGB': self.optimizer_GB.state_dict(),
                    'optimDP': self.optimizer_DP.state_dict(),
                    'optimDP_ll': self.optimizer_DP_ll.state_dict(),
                    'optimSeg': self.optimizer_Seg.state_dict()
                    }, 
                    '%s/iter_%s.ckpt' % (self._output_dir, epoch))
        
        if (epoch != 'latest' and epoch != 'Best'):
            with open(self._output_dir+'/checkpoint_names', 'a') as fp:
                fp.writelines('iter_%s.ckpt\n' % (epoch))


    # def load_networks(self, epoch):
    #     write_step
    # #     """Load all the networks from the disk.
    # #     Parameters:
    # #         epoch (int) -- current epoch; used in the file name '%s_net_%s.pth' % (epoch, name)
    # #     """
    # #     for name in self.model_names:
    # #         if isinstance(name, str):
    #         load_path= '%s/epoch_%s.ckpt' % (self._output_dir, epoch)
            # ckpt = torch.load(load_path)
    #             if self._is_training_value and self.opt.pretrained_name is not None:
    #                 load_dir = os.path.join(self.opt.checkpoints_dir, self.opt.pretrained_name)
    #             else:
    #                 load_dir = self.save_dir

    #             load_path = os.path.join(load_dir, load_filename)
    #             net = getattr(self, 'net' + name)
    #             print('loading the model from %s' % load_path)

    #             state_dict = torch.load(load_path, map_location=torch.device('cuda:0'))
    #             if hasattr(state_dict, '_metadata'):
    #                 del state_dict._metadata

    #             # patch InstanceNorm checkpoints prior to 0.4
    #             # for key in list(state_dict.keys()):  # need to copy keys here because we mutate in loop
    #             #    self.__patch_instance_norm_state_dict(state_dict, net, key.split('.'))
    #             net.load_state_dict(state_dict)
#################################################################### train model ########################################################################
    def train_model(self):
        print("training")
        if not os.path.isdir(self._output_dir):
            os.makedirs(self._output_dir)
        writer = SummaryWriter(self._output_dir)

        trainset = UnalignedDataset(self._train_pth)
        train_loader = torch.utils.data.DataLoader(
            trainset,
            batch_size=self._batch_size,
            shuffle=True,
            num_workers=1,
            drop_last=True if self._is_training_value else False,
            pin_memory=True
        )
    
        ################## test files direction keys
        target = ['ct', 'CT']
        source = ['mr', 'MR']
        ################## test files
        source_txt_path = './data/datalist/test_' +source[0]+ '.txt'
        source_test_list = read_lists(source_txt_path)
        target_txt_path = './data/datalist/test_' +target[0]+ '.txt'
        target_test_list = read_lists(target_txt_path)
        data_size = [1, 256, 256]
        label_size = [1, 256, 256]
        # batch_size = 2

        if "ct" in source_txt_path:
            src_Max = 2.368554272081745
            src_Min = -1.763460938640936
        else:  #### mr
            src_Max = 2.6575754759544323
            src_Min = -1.1368891922215185

        if "mr" in target_txt_path:
            tgt_Max = 2.6575754759544323
            tgt_Min = -1.1368891922215185
            tgt_display_idx = np.array([[2, 50],
                                        [1, 83]])
        else:  #### ct
            tgt_Max = 2.368554272081745
            tgt_Min = -1.763460938640936
            tgt_display_idx = np.array([[3, 140],
                                        [1, 77]])
        total_iters = -1
        epoch = 0
        best_dice=0
        # self.losses_ga = []
        # self.losses_gb = []
        # self.losses_da = []
        # self.losses_db = []
        # self.losses_seg = []
        # self.losses_dpi = []
        self.dice_fake_b = []
        while epoch >= 0:
            epoch_start_time = time.time()
            epoch+=1
########################################### train loop ###############################################
            self.train()
            # reset stuff

            for i, data in enumerate(train_loader):
                # if i > 12:
                #     break
                total_iters += 1  # 0
                self.set_input(data)
                self.optimize_params()
                self.call_last_batch()
                self.num_fake_inputs += 1

                if (total_iters+1) % self.save_interval == 0:              # cache our model every <save_epoch_freq> epochs
                    # for scheduler in self.schedulers:
                    #     # scheduler.step()
                    #     scheduler.step()
                    self.save_networks(total_iters)
                    # self.call_last_batch()
                    with torch.no_grad():
                        # print(self.bit_mask_pred.size())
                        mask = torch.argmax(nn.Softmax(1)(self.A_mask.detach()), 1)
                        prediction = torch.argmax(nn.Softmax(1)(self.pred_fakeB.detach()), 1)
                        rows = []
                        for image_i in range(4):
                            rgb_gt = decode_segmap(mask[image_i])
                            rgb_pred = rgb_pred = decode_segmap(prediction[image_i])
                            pic = [(self.real_A[image_i].repeat(3,1,1)+1)/2, (self.fake_B[image_i].repeat(3,1,1)+1)/2,
                                    rgb_pred, rgb_gt,
                                    (self.bit_mask[image_i].repeat(3,1,1)+1)/2, (torch.nn.functional.interpolate(self.bit_mask_pred, [model.IMG_HEIGHT, model.IMG_WIDTH], mode='area')[image_i].repeat(3,1,1)+1)/2,
                                    (self.real_B[image_i].repeat(3,1,1)+1)/2] # C=3, H, W
                            rows.append(torch.cat(pic, 2)) # C=3, H, W*4

                        pic = torch.cat(rows, 1)  # C=3, H*4, W*4
                        # write_step = (total_iters+1)/self.save_interval
                        writer.add_image('pred', pic, global_step=total_iters, dataformats='CHW')
                        del pic
                        rows.clear()

                iter = (i+1)*self._batch_size
                # if (total_iters+1) % self.print_freq == 0: 
                writer.add_scalar('Fake B dice',
                                    self.dice_fake_b,
                                    total_iters)
                writer.add_scalar('B dice',
                                    self.dice_b,
                                    total_iters)
                    # self.dice_fake_b.clear()
                writer.add_scalar('GA loss',
                                self.losses_ga,
                                total_iters)
                # self.losses_ga.clear()
                writer.add_scalar('GB loss',
                                self.losses_gb,
                                total_iters)
                writer.add_scalar('lambda',
                                torch.mean(self.bit_mask88),
                                total_iters)
                # self.losses_gb.clear()
                writer.add_scalar('DA loss',
                                self.losses_da,
                                total_iters)
                # self.losses_da.clear()
                writer.add_scalar('DB loss',
                                self.losses_db,
                                total_iters)
                writer.add_scalar('DB aux loss',
                                self.losses_db_aux,
                                total_iters)
                # self.losses_db.clear()
                writer.add_scalar('Seg loss',
                                self.losses_seg,
                                total_iters)
                # self.losses_seg.clear()
                writer.add_scalar('DPI loss',
                                self.losses_dpi,
                                total_iters)
                writer.add_scalar('lr',
                                self.optimizer_DA.param_groups[0]['lr'],
                                total_iters)
                # self.losses_dpi.clear()
                t_comp = (time.time()-epoch_start_time)/(i+1)
                print('(epoch: %d, iter: %d, img: (%d/%d), time: %.2f) ' % (epoch, int(i+1), iter, len(trainset), t_comp))
            
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

                writer.add_scalar('PSNR',
                                np.mean(psnr),
                                total_iters)
                psnr.clear()
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

                dice_avg = np.mean(dice)
                assd_avg = np.mean(assd)

                row1 = [(visual_data2.repeat(3,1,1)+1)/2, visual_pred2, visual_gt2]
                row1 = torch.cat(row1, 2)
                row2 = [(visual_data1.repeat(3,1,1)+1)/2, visual_pred1, visual_gt1]
                row2 = torch.cat(row2, 2)

                pic = torch.cat((row1,row2), 1)  # C=3, H*2, W*4
                writer.add_image('Test pred', pic, global_step=total_iters, dataformats='CHW')

                writer.add_scalar('Test seg loss',
                                np.mean(test_seg_losses),
                                total_iters)
                test_seg_losses.clear()
                writer.add_scalar('Test dice',
                                dice_avg,
                                total_iters)
                writer.add_scalar('Test assd',
                                assd_avg,
                                total_iters)

            self.train()
            print('saving the model at the end of epoch %d, iters %d' % (epoch, iter))
            self.save_networks('latest')

            if best_dice < dice_avg:
                self.save_networks('Best')
                best_dice = dice_avg
            print('End of epoch %d: Time Taken: %s' % (epoch, time.strftime("%H:%M:%S",time.gmtime(time.time()-epoch_start_time))))




if __name__ == '__main__':
    manual_seed=1234
    random.seed(manual_seed)
    np.random.seed(manual_seed)
    torch.manual_seed(manual_seed)
    torch.cuda.manual_seed_all(manual_seed)

    with open('config_param.json') as config_file:
        config = json.load(config_file)

    sifa_model = SIFA_pytorch(config)
    sifa_model.train_model()