import torch
import torch.nn as nn

from torch_timeseries.normalizations import *


class Model(nn.Module):
    def __init__(self, f_model_type, forecast_model: nn.Module, norm_model: nn.Module):
        super().__init__()
        self.f_model_type = f_model_type
        self.fm = forecast_model
        self.nm = norm_model

    
    def normalize(self, batch_x, batch_x_enc=None, dec_inp=None,dec_inp_enc=None):
        # normalize
        # input: B T N
        # output: B, T, N
        dec_inp = dec_inp
        if isinstance(self.nm, RevIN):
            batch_x, dec_inp = self.nm(batch_x, 'n', dec_inp)  if 'former' in self.f_model_type  else  self.nm(batch_x)
        elif  "SAN" in self.nm.__class__.__name__: #isinstance(self.nm, SAN):
            batch_x, self.pred_stats = self.nm(batch_x) 
        elif  isinstance(self.nm, DishTS):
            batch_x, dec_inp =self.nm(batch_x, 'n', dec_inp)  if 'former' in self.f_model_type  else  self.nm(batch_x)
        elif  isinstance(self.nm, No):
            pass
        else:
            batch_x = self.nm(batch_x)
        
        return batch_x, dec_inp
            
    def denormalize(self, pred):
        # denormalize
        if isinstance(self.nm, RevIN):
            pred = self.nm(pred, 'd') 
        elif isinstance(self.nm, No):
            pass
        elif  "SAN" in self.nm.__class__.__name__: #isinstance(self.nm, SAN):
            pred = self.nm(pred, 'd', self.pred_stats)
        elif  isinstance(self.nm, DishTS):
            pred = self.nm(pred, 'd') 
        else:
            pred = self.nm(pred, 'd')
        
        return pred
    
    
    def forward(self, batch_x, batch_x_enc=None, dec_inp=None,dec_inp_enc=None):
        
        if 'former' in self.f_model_type:
            pred = self.fm(batch_x, batch_x_enc, dec_inp, dec_inp_enc)
        else:
            pred = self.fm(batch_x)

        return pred

