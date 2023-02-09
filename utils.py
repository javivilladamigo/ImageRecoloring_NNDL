import matplotlib.pyplot as plt
import torch
import numpy as np
from skimage.color import rgb2lab, lab2rgb
import os


class AverageMeter:
    def __init__(self):
        self.reset()
        
    def reset(self):
        self.count, self.avg, self.sum = [0.] * 3
    
    def update(self, val, count=1):
        self.count += count
        self.sum += count * val
        self.avg = float(self.sum / self.count)

def create_loss_meters():
    loss_D_fake = AverageMeter()
    loss_D_real = AverageMeter()
    loss_D = AverageMeter()
    loss_G_GAN = AverageMeter()
    loss_G_L1 = AverageMeter()
    loss_G = AverageMeter()
    
    return {'loss_D_fake': loss_D_fake,
            'loss_D_real': loss_D_real,
            'loss_D': loss_D,
            'loss_G_GAN': loss_G_GAN,
            'loss_G_L1': loss_G_L1,
            'loss_G': loss_G}

def update_losses(model, loss_meter_dict, count):
    for loss_name, loss_meter in loss_meter_dict.items():
        loss = getattr(model, loss_name)
        loss_meter.update(loss.item(), count=count)
        
                   
def my_lab_to_rgb(L, ab):
    '''
        Takes a L and ab channels of a batch of images 
        and converts them into RGB
        Parameters:
        * L  (1-channel tensor array): L channel images (range: [-1, 1], torch tensor array)
        * ab (2-channel tensor array):  ab channel images (range: [-1, 1], torch tensor array)
        Returns:
        * rgb (RGB numpy image): rgb output images  (range: [0, 255], numpy array)
    '''
     
    
    L = (L + 1.) * 50.
    ab = ab * 110.
    
    Lab = torch.cat([L, ab], dim=1).permute(0, 2, 3, 1).cpu().numpy()
    
    rgb_imgs = []
    
    for img in Lab:
        img_rgb = lab2rgb(img)#
        rgb_imgs.append(img_rgb)
    return np.stack(rgb_imgs, axis=0)


def visualize(model, data, epoch, sample, out_path, save=True):

    model.net_G.eval()

    with torch.no_grad():
        model.setup_input(data)
        model.forward()
        
    model.net_G.train()
    fake_color = model.fake_color.detach()

    real_color = model.ab
    L = model.L
  
    ab = model.fake_color
    ab_true = real_color

    fake_imgs = my_lab_to_rgb(L, fake_color)
    real_imgs = my_lab_to_rgb(L, real_color)

    #fig = plt.figure(figsize=(15,15))
    fig, ax = plt.subplots(6, 5, figsize=(12,14))
    
    for i in range(5):
        ## True L channel
        ax[0][i].imshow(L[i][0].cpu(), cmap='gray')
        ax[0][i].set_title('Original L channel', fontsize=12)
        
        ## True color image
        ax[1][i].imshow(real_imgs[i])
        ax[1][i].set_title('Original Image',fontsize=12)
        
        ## Reco color image
        ax[2][i].imshow(fake_imgs[i])
        ax[2][i].set_title('Colorized output',fontsize=12)
        
        ## True L channel
        ax[3][i].imshow(L[i+5][0].cpu(), cmap='gray')
        ax[3][i].set_title('Original L channel', fontsize=12)
        
        ## True color image
        ax[4][i].imshow(real_imgs[i+5])
        ax[4][i].set_title('Original Image',fontsize=12)
        
        ## Reco color image
        ax[5][i].imshow(fake_imgs[i+5])
        ax[5][i].set_title('Colorized output',fontsize=12)
        
        for axs in fig.get_axes():
            axs.axis('off')
        
    fig.tight_layout()
    plt.show()

    if save:
        fig.savefig(f'{out_path}/colorization_ep{epoch}_it{sample}.pdf', dpi = 200)


def log_results(loss_meter_dict):
    for loss_name, loss_meter in loss_meter_dict.items():
        print(f'{loss_name}: {loss_meter.avg:.5f}')

def save_losses(out_path, filename, loss_dict, epoch):
    f_out = out_path + filename
    f = open(f_out, 'a')
    for k, v in loss_dict.items():
        f.write(f'ep {epoch} - {k}: {v.avg:.5f} \n')
    f.close()



class Plotter_GAN(object):
    '''
        Plot loss for G and D with Training and Validation,
    '''

    def __init__(self):
        self.loss_D_fake = []
        self.loss_D_real = []
        self.loss_D = []
        self.loss_G_GAN = []
        self.loss_G_L1 = []
        self.loss_G = []


    def dis_update(self, loss_fake, loss_real, loss):
        if type(loss_fake) != float:
            loss_fake = float(loss_fake)
        if type(loss_real) != float:
            loss_real = float(loss_real)
        if type(loss) != float:
            loss = float(loss)

        self.loss_D_fake.append(loss_fake)
        self.loss_D_real.append(loss_real)
        self.loss_D.append(loss)

    def gen_update(self, loss_gan, loss_l1, loss):
        if type(loss_gan) != float:
            loss_gan = float(loss_gan)
        if type(loss_l1) != float:
            loss_l1 = float(loss_l1)
        if type(loss) != float:
            loss = float(loss)

        self.loss_G_GAN.append(loss_gan)
        self.loss_G_L1.append(loss_l1)
        self.loss_G.append(loss)


    def draw(self, out_path):
        
        fig, ax = plt.subplots(1, 2, figsize=(12,6))
        
        ax[0].plot(self.loss_G, label='Total generator loss')
        ax[1].plot(self.loss_D_fake, label='Discriminator loss - fake')
        ax[1].plot(self.loss_D_real, label='Discriminator loss - real')
        ax[1].plot(self.loss_D, label = 'Total discriminator loss')
        
        
        for axs in fig.get_axes():
            axs.legend(loc='best',frameon=False, fontsize=12, ncol=1)
            axs.set_xlabel('Epoch', fontsize = 12)
            axs.set_ylabel('Loss', fontsize = 12)

        fig.tight_layout()
        
        if not os.path.isdir('loss'):
            os.mkdir('loss')
            
        fig.savefig(f'{out_path}loss/losses.png', dpi = 200)
        plt.clf()
        plt.close()
