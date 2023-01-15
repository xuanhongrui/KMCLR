import numpy as np
from numpy import random
import torch as t
import Kg_Par
import Kg_Data
import Kg_Model
from contrast import Contrast
from torch import optim
import utils
import Mul_Model

seed = 2020
def set_seed(seed):
    np.random.seed(seed)
    if t.cuda.is_available():
        t.cuda.manual_seed(seed)
        t.cuda.manual_seed_all(seed)
    t.manual_seed(seed)

if __name__ == '__main__':

    set_seed(seed)
    dataset = Kg_Data.UILoader(path='./datasets/' + Kg_Par.dataset)
    kg_dataset = Kg_Data.KGDataset(dataset.m_item)
    Kg_model = Kg_Model.Model(Kg_Par.config, dataset, kg_dataset)
    Kg_model = Kg_model.to(Kg_Par.device)
    contrast_model = Contrast(Kg_model).to(Kg_Par.device)
    optimizer = optim.Adam(Kg_model.parameters(), lr=Kg_Par.config['lr'])
    bpr = utils.BPRLoss(Kg_model, optimizer)
    Mul_model = Mul_Model.Model()
    Mul_model.run(Kg_model, contrast_model, optimizer, bpr)

