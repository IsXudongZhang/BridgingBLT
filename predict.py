import numpy as np
import pandas as pd
import torch
from torch.utils import data

torch.manual_seed(2)  # reproducible torch:2 np:3
np.random.seed(3)
from argparse import ArgumentParser
from stream import *
from models_bilstm import BIN_Interaction_Flat
use_cuda = torch.cuda.is_available()
device = torch.device("cuda:0" if use_cuda else "cpu")

parser = ArgumentParser(description='BridgingBLT Predicting.')
parser.add_argument('-b', '--batch-size', default=32, type=int,metavar='N')
parser.add_argument('-j', '--workers', default=0, type=int, metavar='N')

 
class Test_BIN_Data_Encoder(data.Dataset):

    def __init__(self, list_IDs, df_dti):
        'Initialization'
        self.list_IDs = list_IDs
        self.df = df_dti
        
    def __len__(self):
        'Denotes the total number of samples'
        return len(self.list_IDs)

    def __getitem__(self, index):
        index = self.list_IDs[index]

        drug = self.df.iloc[index]['SMILES']
        protein = 'MSHHWGYGKHNGPEHWHKDFPIAKGERQSPVDIDTHTAKYDPSLKPLSVSYDQATSLRILNNGHAFNVEFDDSQDKAVLKGGPLDGTYRLIQFHFHWGSLDGQGSEHTVDKKKYAAELHLVHWNTKYGDFGKAVQQPDGLAVLGIFLKVGSAKPGLQKVVDVLDSIKTKGKSADFTNFDPRGLLPESLDYWTYPGSLTTPPLLECVTWIVLKEPISVSSEQVLKFRKLNFNGEGEPEELMVDNWRPAQPLKNRQIKASFK'
        d_v, input_mask_d = drug2emb_encoder(drug)
        p_v, input_mask_p = protein2emb_encoder(protein)

        return drug, protein, d_v, p_v, input_mask_d, input_mask_p


def test(data_generator, model):
    y_pred = []
    y_drug = []
    model.eval()
    for i, (drug, protein, d, p, d_mask, p_mask) in enumerate(data_generator):
        score = model(d.long().cuda(), p.long().cuda(), d_mask.long().cuda(), p_mask.long().cuda())

        m = torch.nn.Sigmoid()
        logits = torch.squeeze(m(score))

        logits = logits.detach().cpu().numpy()

        y_pred = y_pred + logits.flatten().tolist()
  
        y_drug = list(drug) + y_drug

    y_drug_df = pd.DataFrame(y_drug)
    y_drug_df.to_csv('case_dataset/case_drug.csv', index=False)
    y_pred_df = pd.DataFrame(y_pred)
    y_pred_df.to_csv('case_dataset/y_pred.csv', index=False)



    return

def main():
    args = parser.parse_args()

    model = torch.load('case_dataset/best_model.pth')

    model = model.cuda()

    params = {'batch_size': args.batch_size,
              'shuffle': False,
              'num_workers': args.workers,
              'drop_last': True}

    df_test = pd.read_csv('case_dataset/input_drug.csv')
   
    testing_set = Test_BIN_Data_Encoder(df_test.index.values, df_test)
   
    testing_generator = data.DataLoader(testing_set, **params)

    print('--- Go for Predicting ---')
    with torch.set_grad_enabled(False):
        test(testing_generator, model)

main()
print("Done!")
