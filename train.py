
import time
import os
import glob
import torch
import random
import numpy as np
from PIL import Image
from torch.utils.tensorboard import SummaryWriter
from options.train_options import TrainOptions
from data import create_dataset
from models import create_model
from util.visualizer import Visualizer
from data.base_dataset import get_transform


def set_seed(Seed=123):
    random.seed(Seed)
    np.random.seed(Seed)
    torch.manual_seed(Seed)
    torch.cuda.manual_seed(Seed)
    torch.use_deterministic_algorithms(True, warn_only=True)
    torch.backends.cudnn.benchmark = False


if __name__ == '__main__':
    opt = TrainOptions().parse()

    os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':4096:8'
    Seed = int(opt.seed)
    set_seed(Seed)

    dataset = create_dataset(opt)
    dataset_size = len(dataset)
    print('The number of training images = %d' % dataset_size)

    # get validation dataset
    params_Bv = dict()
    params_Bv['crop_pos'] = (15, 15)
    params_Bv['flip'] = False
    params_Bv['colorjitter'] = False
    params_Bv['new_w'] = opt.load_size
    params_Bv['new_h'] = opt.load_size
    transform_B = get_transform(opt, params=params_Bv, grayscale=(opt.output_nc == 1), normalize=True)
    transform_B_labelmask = get_transform(opt, params=params_Bv, grayscale=(opt.output_nc == 1), normalize=False)
    realdata_dir = 'lesion_detection/datasets'
    style_data = 'liver'
    B_valid_paths = []

    # Need to specify the case IDs in the validation dataset
    valid_style_data = ('case_ID#1', 'case_ID#2')


    for id in valid_style_data:
        B_valid_paths += sorted(glob.glob(os.path.join(realdata_dir, style_data, id, 'images/*.png')))
    B_valids = []
    B_valid_labels = []
    B_valid_masks = []
    for idx, B_path in enumerate(B_valid_paths):
        B_valid_path = B_valid_paths[idx]
        B_valid_path_list = B_valid_path.split('/')
        B_valid_label_path = os.path.join('/',B_valid_path_list[1], B_valid_path_list[2], B_valid_path_list[3], B_valid_path_list[4], B_valid_path_list[5], B_valid_path_list[6], B_valid_path_list[7], 'labels', B_valid_path_list[9].split('.')[0] + '_label.' + B_valid_path_list[9].split('.')[1])
        B_valid_mask_path = os.path.join('/',B_valid_path_list[1], B_valid_path_list[2], B_valid_path_list[3], B_valid_path_list[4], B_valid_path_list[5], B_valid_path_list[6], B_valid_path_list[7], 'masks', B_valid_path_list[9].split('.')[0] + '_mask.' + B_valid_path_list[9].split('.')[1])
        B_valid_img_PIL = Image.open(B_valid_path).convert('L') #$
        B_valid_label_PIL = Image.open(B_valid_label_path).convert('L') #$
        B_valid_mask_PIL = Image.open(B_valid_mask_path).convert('L') #$
        B_valids.append(transform_B(B_valid_img_PIL).unsqueeze(0))
        B_valid_labels.append((transform_B_labelmask(B_valid_label_PIL)>0.5).long().unsqueeze(0))
        B_valid_masks.append((transform_B_labelmask(B_valid_mask_PIL)>0.5).long().unsqueeze(0))

    model = create_model(opt)
    model.setup(opt)
    visualizer = Visualizer(opt)
    total_iters = 0
    ep_pd1 = 35
    ep_pd2 = 60

    # logger
    logPath = 'lesion_detection/DA/rgGAN/log/'+opt.model_name+'_lbd'+str(int(opt.lambda_tumor))+'_run'+str(opt.run_number)
    writer = SummaryWriter(logPath)

    valid_vals_EMA, coef_EMA, count_EMA = [], 0.9, 1
    best_score = 10000.0
    best_score_EMA = 10000.0
    tolerance = 5000
    tol_total_iters = 99999999

    for epoch in range(opt.epoch_count, opt.n_epochs + opt.n_epochs_decay + 1):
        epoch_start_time = time.time()
        iter_data_time = time.time()
        epoch_iter = 0
        visualizer.reset()
        model.update_learning_rate()
        for i, data in enumerate(dataset):
            iter_start_time = time.time()
            if total_iters % opt.print_freq == 0:
                t_data = iter_start_time - iter_data_time

            total_iters += opt.batch_size
            epoch_iter += opt.batch_size
            model.set_input(data)

            if epoch <= ep_pd1:
                model.optimize_parameters_GD()
            elif epoch <= ep_pd2:
                model.optimize_parameters_LD()

            else:
                model.optimize_parameters_all()

            if total_iters % opt.print_freq == 0 and total_iters>0:
                losses = model.get_current_losses()
                t_comp = (time.time() - iter_start_time) / opt.batch_size
                visualizer.print_current_losses(epoch, epoch_iter, losses, t_comp, t_data)
                for k, v in losses.items():
                    writer.add_scalar(k, v, total_iters)

                # validation
                if epoch > ep_pd1:
                    loss_LD_valid = model.valid_LD_GA(valid_Bs=B_valids, valid_B_labels=B_valid_labels, valid_B_masks=B_valid_masks)
                    writer.add_scalar('LD_valid', loss_LD_valid, total_iters)
                    if not valid_vals_EMA:
                        valid_val_cur = 0.0
                    valid_val_cur = coef_EMA * valid_val_cur + (1 - coef_EMA) * loss_LD_valid
                    valid_val_cur_biascorr = valid_val_cur / (1 - coef_EMA ** count_EMA)
                    valid_vals_EMA.append(valid_val_cur_biascorr)
                    count_EMA += 1

                    print('\nValidation loss-EMA: {0}, best score-EMA: {1}'.format(valid_val_cur_biascorr, best_score_EMA))
                    if valid_val_cur_biascorr <=  best_score_EMA:
                        best_score_EMA = valid_val_cur_biascorr
                        print('update to new best_score_EMA: {}'.format(best_score_EMA))
                        model.save_networks('best_EMA')
                        print('Save best weights to: best_EMA*.pth')
                        count_ = 0
                    else:
                        count_ = count_ + 1
                    print('\nValidation loss: {}, best_score: {}'.format(loss_LD_valid, best_score))
                    if loss_LD_valid <=  best_score:
                        best_score = loss_LD_valid
                        print('update to new best_score: {}'.format(best_score))
                        model.save_networks('best')
                        print('Save best weights to: best*.pth')

                    if count_ >= tolerance:
                        if tol_total_iters > total_iters:
                            tol_total_iters = total_iters
                        print('performance not imporoved for so long, since ', tol_total_iters)

            if total_iters % opt.save_latest_freq == 0:
                print('saving the latest model (epoch %d, total_iters %d)' % (epoch, total_iters))
                save_suffix = 'iter_%d' % total_iters if opt.save_by_iter else 'latest'
                model.save_networks(save_suffix)

            iter_data_time = time.time()
        if epoch % opt.save_epoch_freq == 0:
            print('saving the model at the end of epoch %d, iters %d' % (epoch, total_iters))
            model.save_networks('latest')
            model.save_networks(epoch)

        print('End of epoch %d / %d \t Time Taken: %d sec' % (epoch, opt.n_epochs + opt.n_epochs_decay, time.time() - epoch_start_time))

    writer.close()
