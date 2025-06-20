import torch
import itertools
from util.image_pool import ImagePool
from .base_model import BaseModel
from . import networks

import click
from .models import models
from .models import get_LD_model
from .util import weighted_loss2D
from .torch_utils import to_device
import os

class RGGANS2Model(BaseModel):

    @staticmethod
    def modify_commandline_options(parser, is_train=True):
        parser.set_defaults(no_dropout=True)
        if is_train:
            parser.add_argument('--lambda_A', type=float, default=10.0, help='weight for cycle loss (A -> B -> A)')
            parser.add_argument('--lambda_B', type=float, default=10.0, help='weight for cycle loss (B -> A -> B)')
            parser.add_argument('--lambda_identity', type=float, default=0.5, help='use identity mapping. Setting lambda_identity other than 0 has an effect of scaling the weight of the identity mapping loss. For example, if the weight of the identity loss should be 10 times smaller than the weight of the reconstruction loss, please set lambda_identity = 0.1')
            parser.add_argument('--model_LD', default='unet2d', type=click.Choice(models.keys()))
            parser.add_argument('--momentum', '-m', default=0.99)

        return parser

    def __init__(self, opt):
        BaseModel.__init__(self, opt)
        self.loss_names = ['D_A', 'G_A', 'cycle_A', 'idt_A', 'D_B', 'G_B', 'cycle_B', 'idt_B', 'tumor_A']
        visual_names_A = ['real_A', 'fake_B', 'rec_A']
        visual_names_B = ['real_B', 'fake_A', 'rec_B']
        if self.isTrain and self.opt.lambda_identity > 0.0:
            visual_names_A.append('idt_B')
            visual_names_B.append('idt_A')

        self.visual_names = visual_names_A + visual_names_B
        if self.isTrain:
            self.model_names = ['G_A', 'G_B', 'D_A', 'D_B', 'LD_GA']
        else:
            self.model_names = ['G_A', 'G_B', 'LD_GA']

        self.netG_A = networks.define_G(opt.input_nc, opt.output_nc, opt.ngf, opt.netG, opt.norm,
                                        not opt.no_dropout, opt.init_type, opt.init_gain, self.gpu_ids)
        self.netG_B = networks.define_G(opt.output_nc, opt.input_nc, opt.ngf, opt.netG, opt.norm,
                                        not opt.no_dropout, opt.init_type, opt.init_gain, self.gpu_ids)

        if self.isTrain:
            self.netD_A = networks.define_D(opt.output_nc, opt.ndf, opt.netD,
                                            opt.n_layers_D, opt.norm, opt.init_type, opt.init_gain, self.gpu_ids)
            self.netD_B = networks.define_D(opt.input_nc, opt.ndf, opt.netD,
                                            opt.n_layers_D, opt.norm, opt.init_type, opt.init_gain, self.gpu_ids)
            self.netLD_GA = get_LD_model(opt.model_LD, finetune=True)


        if self.isTrain:
            if opt.lambda_identity > 0.0:
                assert(opt.input_nc == opt.output_nc)
            self.fake_A_pool = ImagePool(opt.pool_size)
            self.fake_B_pool = ImagePool(opt.pool_size)

            self.criterionGAN = networks.GANLoss(opt.gan_mode).to(self.device)
            self.criterionCycle = torch.nn.L1Loss()
            self.criterionIdt = torch.nn.L1Loss()
            self.criterionFt = torch.nn.MSELoss()

            self.optimizer_G = torch.optim.Adam(itertools.chain(self.netG_A.parameters(), self.netG_B.parameters()), lr=opt.lr, betas=(opt.beta1, 0.999))
            self.optimizer_D = torch.optim.Adam(itertools.chain(self.netD_A.parameters(), self.netD_B.parameters()), lr=opt.lr, betas=(opt.beta1, 0.999))
            self.optimizer_LD_GA = torch.optim.SGD(itertools.chain(self.netLD_GA.parameters()), lr=opt.lr_LD, momentum=opt.momentum, nesterov = True, weight_decay=1e-06)
            self.optimizers.append(self.optimizer_G)
            self.optimizers.append(self.optimizer_D)


    def set_input(self, input):
        AtoB = self.opt.direction == 'AtoB'
        self.real_A = input['A' if AtoB else 'B'].to(device=self.device, dtype=torch.float)
        self.real_A_label = input['A_label'].to(self.device)
        self.real_A_mask = input['A_mask'].to(self.device)
        self.A_cycle_mask = input['A_cycle_mask'].to(self.device)
        self.real_B = input['B' if AtoB else 'A'].to(device=self.device, dtype=torch.float)
        self.image_paths = input['A_paths' if AtoB else 'B_paths']

    def forward(self):
        self.fake_B = self.netG_A(self.real_A)
        self.rec_A = self.netG_B(self.fake_B)
        self.fake_A = self.netG_B(self.real_B)
        self.rec_B = self.netG_A(self.fake_A)

        self.LD_GA, _ = self.netLD_GA(self.real_A)

    def backward_D_basic(self, netD, real, fake):
        
        pred_real = netD(real)
        loss_D_real = self.criterionGAN(pred_real, True)
        
        pred_fake = netD(fake.detach())
        loss_D_fake = self.criterionGAN(pred_fake, False)
        
        loss_D = (loss_D_real + loss_D_fake) * 0.5
        loss_D.backward()
        return loss_D

    def backward_D_A(self):
        fake_B = self.fake_B_pool.query(self.fake_B)
        self.loss_D_A = self.backward_D_basic(self.netD_A, self.real_B, fake_B)

    def backward_D_B(self):
        fake_A = self.fake_A_pool.query(self.fake_A)
        self.loss_D_B = self.backward_D_basic(self.netD_B, self.real_A, fake_A)

    def backward_G(self):
        lambda_idt = self.opt.lambda_identity
        lambda_A = self.opt.lambda_A
        lambda_B = self.opt.lambda_B
        # Identity loss
        if lambda_idt > 0:
            self.idt_A = self.netG_A(self.real_B)
            self.loss_idt_A = self.criterionIdt(self.idt_A, self.real_B) * lambda_B * lambda_idt
            self.idt_B = self.netG_B(self.real_A)
            self.loss_idt_B = self.criterionIdt(self.idt_B, self.real_A) * lambda_A * lambda_idt
        else:
            self.loss_idt_A = 0
            self.loss_idt_B = 0

        self.loss_G_A = self.criterionGAN(self.netD_A(self.fake_B), True)
        self.loss_G_B = self.criterionGAN(self.netD_B(self.fake_A), True)
        self.loss_cycle_A = self.criterionCycle(self.rec_A * self.A_cycle_mask, self.real_A * self.A_cycle_mask) * lambda_A
        self.loss_cycle_B = self.criterionCycle(self.rec_B, self.real_B) * lambda_B
        self.loss_G = self.loss_G_A + self.loss_G_B + self.loss_cycle_A + self.loss_cycle_B + self.loss_idt_A + self.loss_idt_B
        self.loss_G.backward()
        
        self.loss_tumor_A = weighted_loss2D(self.real_A_label, self.LD_GA, annotate_mask = self.real_A_mask, weight=5.0)

    def backward_G_wLD(self):
        lambda_idt = self.opt.lambda_identity
        lambda_A = self.opt.lambda_A
        lambda_B = self.opt.lambda_B
        lambda_tumor = self.opt.lambda_tumor
        
        if lambda_idt > 0:
            self.idt_A = self.netG_A(self.real_B)
            self.loss_idt_A = self.criterionIdt(self.idt_A, self.real_B) * lambda_B * lambda_idt
            self.idt_B = self.netG_B(self.real_A)
            self.loss_idt_B = self.criterionIdt(self.idt_B, self.real_A) * lambda_A * lambda_idt
        else:
            self.loss_idt_A = 0
            self.loss_idt_B = 0

        self.loss_G_A = self.criterionGAN(self.netD_A(self.fake_B), True)
        self.loss_G_B = self.criterionGAN(self.netD_B(self.fake_A), True)
        self.loss_cycle_A = self.criterionCycle(self.rec_A * self.A_cycle_mask, self.real_A * self.A_cycle_mask) * lambda_A
        self.loss_cycle_B = self.criterionCycle(self.rec_B, self.real_B) * lambda_B
        
        lambda_tumor = self.opt.lambda_tumor
        self.LD_GA, self.LD_GA_ft = self.netLD_GA(self.fake_B.detach())
        self.loss_tumor_GA = weighted_loss2D(self.real_A_label, self.LD_GA, annotate_mask = self.real_A_mask, weight=5.0)
        self.loss_Tm =  self.loss_tumor_GA * lambda_tumor
        self.loss_G = self.loss_G_A + self.loss_G_B + self.loss_cycle_A + self.loss_cycle_B + self.loss_idt_A + self.loss_idt_B + self.loss_Tm
        self.loss_G.backward()
    
    def backward_LD_GA(self):
        lambda_tumor = self.opt.lambda_tumor
        self.LD_GA, _ = self.netLD_GA(self.real_A.detach())
        self.loss_tumor_A = weighted_loss2D(self.real_A_label, self.LD_GA, annotate_mask = self.real_A_mask, weight=5.0)

        self.loss_LD = self.loss_tumor_A
        self.loss_LD.backward()
        


    def valid_LD_GA(self, valid_Bs, valid_B_labels, valid_B_masks):
        running_valid_loss = 0.0
        for i in range(len(valid_Bs)):
            LD_GA_valid = self.netLD_GA.predict(valid_Bs[i])
            loss_tumor_valid_GA = weighted_loss2D(valid_B_labels[i].cuda(), torch.from_numpy(LD_GA_valid).cuda(), annotate_mask = valid_B_masks[i].cuda(), weight=5.0)
            loss_LD_valid = loss_tumor_valid_GA.data.cpu().numpy().mean()
            running_valid_loss += loss_LD_valid
        running_valid_loss /= len(valid_Bs)
        return running_valid_loss

    def optimize_parameters_GD(self):
        
        self.forward()
        self.set_requires_grad([self.netD_A, self.netD_B], False)
        self.optimizer_G.zero_grad()
        self.backward_G()
        self.optimizer_G.step()
        
        self.set_requires_grad([self.netD_A, self.netD_B], True)
        self.optimizer_D.zero_grad()
        self.backward_D_A()
        self.backward_D_B()
        self.optimizer_D.step()
    
    def optimize_parameters_LD(self):
        self.forward()
        self.set_requires_grad([self.netD_A, self.netD_B, self.netG_A, self.netG_B], False)
        self.set_requires_grad([self.netLD_GA], True)
        self.optimizer_LD_GA.zero_grad()
        self.backward_LD_GA()
        self.optimizer_LD_GA.step()
        self.loss_idt_A = 0
        self.loss_idt_B = 0
        self.loss_G_A = 0
        self.loss_G_B = 0
        self.loss_cycle_A = 0
        self.loss_cycle_B = 0
        self.loss_D_A = 0
        self.loss_D_B = 0
        
    def optimize_parameters_all(self):

        self.forward()
        self.set_requires_grad([self.netD_A, self.netD_B], False)
        self.set_requires_grad([self.netLD_GA], False)
        self.set_requires_grad([self.netG_A, self.netG_B], True)
        self.optimizer_G.zero_grad()
        self.backward_G_wLD()
        self.optimizer_G.step()
        self.set_requires_grad([self.netD_A, self.netD_B], True)
        self.optimizer_D.zero_grad()
        self.backward_D_A()
        self.backward_D_B()
        self.optimizer_D.step()
        self.set_requires_grad([self.netLD_GA], True)
        self.optimizer_LD_GA.zero_grad()
        self.backward_LD_GA()
        self.optimizer_LD_GA.step()


