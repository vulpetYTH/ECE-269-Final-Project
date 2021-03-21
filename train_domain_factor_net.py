import os
from os.path import join

# Import from torch
import torch
import torch.optim as optim

# Import from within Package 
from ..models.utils import get_model
from ..data.utils import load_data_multi

import pdb


def soft_cross_entropy(input, target, size_average=True):
    logsoftmax = torch.nn.LogSoftmax(dim=1)
    if size_average:
        return torch.mean(torch.sum(-target * logsoftmax(input), dim=1))
    else:
        return torch.sum(torch.sum(-target * logsoftmax(input), dim=1))


def train_epoch(loader_src, loader_tgt, net, opt_domain_factor, opt_decoder, opt_dis_cls, epoch,
                gamma_dispell, gamma_rec, num_cls, fake_label_type):
   
    log_interval = 10  # specifies how often to display
  
    N = min(len(loader_src.dataset), len(loader_tgt.dataset)) 
    joint_loader = zip(loader_src, loader_tgt)

    # Only make discriminator trainable
    net.discriminator_cls.train()
    net.domain_factor_net.eval()
    # net.domain_factor_net.train()
    net.decoder.eval()
    net.tgt_net.eval()

    last_update = -1
    for batch_idx, ((data_s, cls_s_gt), (data_t, cls_t_gt)) in enumerate(joint_loader):
        
        # log basic mann train info
        info_str = "[Train domain_factor Net] Epoch: {} [{}/{} ({:.2f}%)]".format(epoch, batch_idx*len(data_t),
                                                                          N, 100 * batch_idx / N)
   
        ########################
        # Setup data variables #
        ########################
        if torch.cuda.is_available():
            data_s = data_s.cuda()
            data_t = data_t.cuda()

        data_s.require_grad = False
        data_t.require_grad = False

        ##########################
        # Optimize Discriminator #
        ##########################

        # extract features and logits
        data_cat = torch.cat((data_s, data_t), dim=0).detach()

        with torch.no_grad():  # content branch

            _, content_ftr_s = net.tgt_net(data_s.clone())
            content_ftr_s = content_ftr_s.detach()

            logit_t_pseudo, content_ftr_t = net.tgt_net(data_t.clone())
            logit_t_pseudo = logit_t_pseudo.detach()
            content_ftr_t = content_ftr_t.detach()

            domain_factor_ftr = net.domain_factor_net(data_cat.clone())  # domain_factor branch
            domain_factor_ftr = domain_factor_ftr.detach()

        # predict classes with discriminator using domain_factor feature
        pred_cls_from_domain_factor = net.discriminator_cls(domain_factor_ftr.clone())

        # prepare class labels
        cls_t_pseudo = logit_t_pseudo.argmax(dim=1)
        pseudo_acc = (cls_t_pseudo == cls_t_gt.cuda()).float().mean()  # acc of pseudo label
        info_str += " pseudo_acc: {:0.1f}".format(pseudo_acc.item() * 100)
        cls_real = torch.cat((cls_s_gt.cuda(), cls_t_pseudo), dim=0).cuda()  # real

        # compute loss for class disciminator
        loss_dis_cls = net.gan_criterion(pred_cls_from_domain_factor, cls_real)

        # zero gradients for optimizer
        opt_dis_cls.zero_grad()
        # loss backprop
        loss_dis_cls.backward()
        # optimize discriminator
        opt_dis_cls.step()

        # compute discriminator acc
        pred_dis_cls = torch.squeeze(pred_cls_from_domain_factor.argmax(1))
        acc_cls = (pred_dis_cls == cls_real).float().mean()
        
        # log discriminator update info
        info_str += " D_acc: {:0.1f} D_loss: {:.3f}".format(acc_cls.item()*100, loss_dis_cls.item())

        ##########################
        # Optimize domain_factor Network #
        ##########################

        if acc_cls.item() > 0.3:

            # Make domain_factor net trainable
            net.discriminator_cls.eval()
            # net.discriminator_cls.train()
            net.domain_factor_net.train()
            net.decoder.train()

            # update domain_factor net
            last_update = batch_idx

            ###############
            # GAN loss - domain_factor should not include class information
            # Calculate domain_factors again and predict classes with it
            domain_factor_ftr = net.domain_factor_net(data_cat.clone())
            pred_cls_from_domain_factor = net.discriminator_cls(domain_factor_ftr.clone())

            # Calculate loss using random class labels
            if fake_label_type == 'random':
                cls_fake = torch.randint(0, num_cls, (cls_real.size(0),)).long().cuda()
                loss_gan_domain_factor = net.gan_criterion(pred_cls_from_domain_factor, cls_fake)
            elif fake_label_type == 'uniform':
                cls_fake = torch.ones((cls_real.size(0), num_cls), dtype=torch.float32).cuda() * 1. / num_cls
                loss_gan_domain_factor = soft_cross_entropy(pred_cls_from_domain_factor, cls_fake)
            else:
                raise Exception("No such fake_label_type: {}".format(fake_label_type))

            ###############
            # reconstruction loss - However, domain_factor should be able to help reconstruct the data into domain specific appearences

            # Concate source and target contents
            cls_ftr = torch.cat((content_ftr_s, content_ftr_t), 0).detach()
            # Concate contents and domain_factors of each sample and feed into decoder
            combined_ftr = torch.cat((cls_ftr, domain_factor_ftr), dim=1)

            data_rec = net.decoder(combined_ftr)

            # Calculate reconstruction loss based on the decoder outputs
            loss_rec = net.rec_criterion(data_rec, data_cat)

            loss = gamma_dispell * loss_gan_domain_factor + gamma_rec * loss_rec

            opt_dis_cls.zero_grad()
            opt_domain_factor.zero_grad()
            opt_decoder.zero_grad()
            
            
            
            
            
#            
#            ################################
#            #### peter added
#            
#            if os.path.exists('results/svhn_bal_to_multi/mann/mann_AMN_net_svhn_bal_multi.pth'): 
#                updated_state_dict, net2_weights = load_arch_weights(net.tgt_net.state_dict(), 'results/svhn_bal_to_multi/mann/mann_AMN_net_svhn_bal_multi.pth')
#                net.tgt_net.load_state_dict(updated_state_dict)
#                #print(">>>>>>>>> [CHECK] peter: loading best source model>>>>>>>>")
#            
#            # l2 regularization on interleaving architecture weights
#            for name, param in net.tgt_net.named_parameters():
#                if net2_weights is not None and name in net2_weights:
#                    loss_gan_t += 0.5 * w_weight_decay * torch.pow((param - net2_weights[name]).norm(2), 2)
#                    # loss1 += 0.5 * interleaving_weight_decay * torch.pow((param - net1_weights[name]).norm(2), 2)
#                else:
#                    loss_gan_t += 0.5 * w_weight_decay * torch.pow(param.norm(2), 2)
#            loss_gan_t.backward()
#            # gradient clipping
#            nn.utils.clip_grad_norm_(net.tgt_net.parameters(), w_grad_clip)
#            opt_net.step()
#            save_arch_weights('results/svhn_bal_to_multi/mann/mann_AMN_net_svhn_bal_multi_tgt.pth', net.tgt_net)
#            
#            ####### phase 2
#            updated_state_dict, net1_weights = load_arch_weights(net.src_net.state_dict(), 'results/svhn_bal_to_multi/mann/mann_AMN_net_svhn_bal_multi_tgt.pth')
#            net.src_net.load_state_dict(updated_state_dict)
#            
#            ## peter added
#            opt_src.zero_grad()
#            ##
#            
#            score_s, x_s = net.src_net(data_s.clone())
#            loss_cls = net.src_net.criterion_cls(score_s.clone(), target_s)
#            loss_ctr = net.src_net.criterion_ctr(x_s.clone(), target_s)
#            loss_src = loss_cls + 0.1 * loss_ctr
#            
#            # l2 regularization on interleaving architecture weights
#            for name, param in net.src_net.named_parameters():
#                if net1_weights is not None and name in net1_weights:
#                    loss_src += 0.5 * w_weight_decay * torch.pow((param - net1_weights[name]).norm(2), 2)
#                    # loss1 += 0.5 * interleaving_weight_decay * torch.pow((param - net1_weights[name]).norm(2), 2)
#                else:
#                    loss_src += 0.5 * w_weight_decay * torch.pow(param.norm(2), 2)
#            
#            loss_src.backward()
#            
#            nn.utils.clip_grad_norm_(net.src_net.parameters(), w_grad_clip)
#            opt_src.step()
#            save_arch_weights('results/svhn_bal_to_multi/mann/mann_AMN_net_svhn_bal_multi.pth', net.tgt_net)
#            
            
            
            
            
            

            loss.backward()

            opt_domain_factor.step()
            opt_decoder.step()

            info_str += " G_loss: {:.3f}".format(loss_gan_domain_factor.item())
            info_str += " R_loss: {:.3f}".format(loss_rec.item())

        ###########
        # Logging #
        ###########
        if batch_idx % log_interval == 0:
            print(info_str)

    return last_update


def train_domain_factor_multi(args):

    """Main function for training domain_factor."""

    src = args.src
    tgt = args.tgt
    base_model = args.base_model
    domain_factor_model = args.domain_factor_model
    num_cls = args.num_cls
    tgt_list = args.tgt_list
    num_epoch = args.domain_factor_num_epoch
    batch = args.batch
    datadir = args.datadir
    outdir = args.outdir_domain_factor
    mann_weights = args.mann_net_file
    lr = args.domain_factor_lr
    betas = tuple(args.betas)
    weight_decay = args.weight_decay
    gamma_dispell = args.gamma_dispell
    gamma_rec = args.gamma_rec
    fake_label_type = args.fake_label_type

    ###########################
    # Setup cuda and networks #
    ###########################

    # setup cuda
    if torch.cuda.is_available():
        kwargs = {'num_workers': 8, 'pin_memory': True}
    else:
        kwargs = {}

    # setup network 
    net = get_model('DomainFactorNet', num_cls=num_cls,
                    base_model=base_model, domain_factor_model=domain_factor_model,
                    weights_init=mann_weights)
    
    # print network and arguments
    print(net)
    print('Training domain_factor {} model for {}->{}'.format(domain_factor_model, src, tgt))

    #######################################
    # Setup data for training and testing #
    #######################################

    train_src_data = load_data_multi(src, 'train', batch=batch, 
                                     rootdir=join(datadir, src), num_channels=net.num_channels,
                                     image_size=net.image_size, download=False, kwargs=kwargs)

    train_tgt_data = load_data_multi(tgt_list, 'train', batch=batch, 
                                     rootdir=datadir, num_channels=net.num_channels,
                                     image_size=net.image_size, download=False, kwargs=kwargs)

    ######################
    # Optimization setup #
    ######################
    opt_domain_factor = optim.Adam(net.domain_factor_net.parameters(),
                           lr=lr, weight_decay=weight_decay, betas=betas)
    opt_decoder = optim.Adam(net.decoder.parameters(),
                           lr=lr, weight_decay=weight_decay, betas=betas)
    opt_dis_cls = optim.Adam(net.discriminator_cls.parameters(), lr=lr,
                             weight_decay=weight_decay, betas=betas)

    ##############
    # Train Mann #
    ##############
    for epoch in range(num_epoch):

        err = train_epoch(train_src_data, train_tgt_data, net, opt_domain_factor, opt_decoder, opt_dis_cls,
                          epoch, gamma_dispell, gamma_rec, num_cls, fake_label_type)

    ######################
    # Save Total Weights #
    ######################
    os.makedirs(outdir, exist_ok=True)
    outfile = join(outdir, 'DomainFactorNet_{:s}_net_{:s}_{:s}.pth'.format(
        domain_factor_model, src, tgt))
    print('Saving to', outfile)
    net.save(outfile)

