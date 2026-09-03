from .BaseModel import BaseModel, BaseLoss
from .TarNet import TarNet




class CFRNet(TarNet):
    '''The same as TARNet, but loss is modified to include IPM'''
    pass

def cfrnet_loss(t_pred, y_preds, t_true, y_true, phi_x=None, *_, alpha=1.0, task='regression', ipm_mode='mmd_rbf', ipm_sigma=None):
    return BaseLoss(t_pred=t_pred, y_preds=y_preds,
                       t_true=t_true, y_true=y_true,
                       phi_x=phi_x,
                       loss_type='tarnet',IPM=True,
                       alpha=alpha, task=task,
                       ipm_mode=ipm_mode, ipm_sigma=ipm_sigma)
