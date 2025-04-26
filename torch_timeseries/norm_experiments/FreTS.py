
import fire
import torch
import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm
from torch_timeseries.norm_experiments.experiment import NormExperiment
import torch.nn as nn
from dataclasses import asdict,dataclass
from torch_timeseries.models import FreTS

from dataclasses import dataclass, asdict


@dataclass
class FreTSExperiment(NormExperiment):
    model_type: str = "FreTS"

    individual : bool = False

    lr: float = 0.0001

    def _init_f_model(self):
        self.f_model = FreTS(
            enc_in=self.dataset.num_features,
            seq_len=self.windows,
            pred_len=self.pred_len,
        )
        self.f_model = self.f_model.to(self.device)

    def _process_batch(self, batch_x, batch_y, batch_x_date_enc, batch_y_date_enc):
        # inputs:
        # batch_x: (B, T, N)
        # batch_y: (B, O, N)
        # ouputs:
        # - pred: (B, N)/(B, O, N)
        # - label: (B, N)/(B, O, N)
        batch_x , _ = self.model.normalize(batch_x) # (B, T, N)   # (B,L,N)
        
        pred = self.model.fm(batch_x) # (B, O, N)
        
        pred = self.model.denormalize(pred)

        return pred, batch_y # (B, O, N), (B, O, N)



def main():
    exp = FreTSExperiment(
        dataset_type="DummyContinuous",
        data_path="./data",
        norm_type='No', # No RevIN DishTS SAN 
        optm_type="Adam",
        batch_size=128,
        device="cuda:1",
        windows=96,
        pred_len=96,
        horizon=1,
        epochs=100,
    )
        
    exp.run()
    
    
def cli():
    fire.Fire(FreTSExperiment)
    

if __name__ == "__main__":
    cli()
