import os
# Kaggle T4 x2: expose both GPUs so DataParallel can split each batch.
# (Embedding precomputation with ESM/IgFold still runs on cuda:0.)
os.environ["CUDA_VISIBLE_DEVICES"] = '0,1'
from antigen_antibody_emb import * 
from antibinder_model import *
import torch
import torch.nn as nn 
import numpy as np 
from torch.utils.data import DataLoader 
from copy import deepcopy 
from tqdm import tqdm
import sys 
import argparse
from utils.utils import CSVLogger_my
from sklearn.metrics import accuracy_score, precision_score, f1_score, classification_report, recall_score
sys.path.append ('../') 
import warnings 
warnings.filterwarnings("ignore")


class Trainer():
    def __init__(self,model,train_dataloader,args,logger,load=False) -> None:
        self.model = model
        self.train_dataloader = train_dataloader
        # self.vaLid_dataloader = valid_dataloader
        # self.test_dataloader = test_dataloader
        self.args = args
        self.logger = logger
        # self.grad_clip = args.grad_clip # cLip gradients at this value, or disable if == 0.0
        self.best_loss = None
        self.load = load

        if self.load==False:
            self.init()
        else:
            print("no init model")

    def init(self):
        init = AntiModelIinitial()
        self.model.apply(init._init_weights)
        print("init successfully!")


    def matrix(self,yhat,y):
        return sum (y==yhat)
    

    def matrix_val(self,yhat,y) :
        # print(sum(yhat))
        return accuracy_score(y,yhat), precision_score(y, yhat), f1_score(y,yhat), recall_score(y, yhat)
    

    def train(self, criterion, epochs):
        optimizer = torch.optim.Adam(self.model.parameters(),lr=self.args.lr)
        # Mixed precision: roughly halves activation memory and speeds up T4s.
        scaler = torch.cuda.amp.GradScaler()
        accum = max(1, self.args.grad_accum)
        for epoch in range(epochs) :
            self.model.train(True)
            train_acc = 0
            train_loss = 0
            num_train = 0
            Y_hat = []
            Y = []
            optimizer.zero_grad()
            for step, (antibody_set, antigen_set, label) in enumerate(tqdm(self.train_dataloader)):
                with torch.cuda.amp.autocast():
                    probs = self.model(antibody_set, antigen_set)
                    y = label.float().cuda()
                    # Compute BCE in fp32: Sigmoid+log is numerically unstable in fp16.
                    loss = criterion(probs.float().view(-1), y.view(-1)) / accum

                yhat = (probs > 0.5).long()
                scaler.scale(loss).backward()

                if (step + 1) % accum == 0 or (step + 1) == len(self.train_dataloader):
                    scaler.step(optimizer)
                    scaler.update()
                    optimizer.zero_grad()

                train_loss += loss.item() * accum
                num_train += antibody_set[0].shape[0]
                Y_hat.extend(yhat)
                Y.extend(y)

            train_acc, train_precision, train_f1, recall = self.matrix_val((torch.cat([temp.view(1, -1) for temp in Y_hat], dim=0)).long().cpu().numpy(),
                                                                            torch.tensor(Y))
            train_loss = train_loss / num_train
            train_loss = np.exp(train_loss)

            self.logger.log([epoch+1, train_loss, train_acc, train_precision,train_f1,recall])

            if self.best_loss==None or train_loss < self.best_loss:
                print('epoch: ',epoch, 'saving...')
                self.best_loss = train_loss
                self.save_model()


    def save_model(self):
        # DataParallel wraps the model in .module; save unwrapped weights
        # so checkpoints load directly into antibinder(...) later.
        raw_model = self.model.module if isinstance(self.model, nn.DataParallel) else self.model
        torch.save(raw_model.state_dict(),f"./ckpts/{self.args.model_name}_{self.args.data}_{self.args.batch_size}_{self.args.epochs}_{self.args.latent_dim}_{self.args.lr}.pth")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--seed', type=int, default=42)
    # Global batch size (split across GPUs by DataParallel). 32 = 16 per T4.
    parser.add_argument('--batch_size', type=int, default=32)
    # Effective batch = batch_size * grad_accum (32 * 4 = 128 by default).
    parser.add_argument('--grad_accum', type=int, default=4)
    # Skip the ESM/IgFold embedding precomputation pass (only safe when every
    # embedding is already cached on disk / in LMDB).
    parser.add_argument('--skip_precompute', action='store_true')
    parser.add_argument('--latent_dim', type=int, default=36)
    # In certain datasets, an early stopping strategy is required to achieve optimal results.
    parser.add_argument('--epochs', type=int, default=500)
    # parser.add_argument('--weight_decay', type=float, default=1e-5, help='weight decay used in optimizer') # 1e-5
    parser.add_argument('--lr', type=float, default=6e-5, help='learning rate')
    parser.add_argument('--model_name', type=str, default= 'AntiBinder')
    parser.add_argument('--cuda', type=bool, default=True)
    parser.add_argument('--data', type=str, default='train')
    # Path to the split CSV (must contain H-FR1..H-FR4, vh, Antigen Sequence,
    # ANT_Binding and an "Antigen" id column). Override with --data_path.
    parser.add_argument('--data_path', type=str,
                        default='./datasets/process_data/COVID-19/Cov_with_target_split.csv')
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)


    antigen_config = configuration()
    setattr(antigen_config, 'max_position_embeddings',1024)

    antibody_config = configuration()
    setattr(antibody_config, 'max_position_embeddings',149)

    # here choose dataset
    data_path = args.data_path
    if not os.path.exists(data_path):
        raise FileNotFoundError(
            f"Data CSV not found: {data_path}. "
            f"Pass a valid file with --data_path (cwd={os.getcwd()})"
        )

    # NOTE: the dataset constructor loads ESM-2 650M and 4 IgFold models onto
    # the GPU. Build the dataset FIRST, precompute all embeddings, and release
    # those encoders before allocating the training model.
    train_dataset = antibody_antigen_dataset(antigen_config=antigen_config,antibody_config=antibody_config,data_path=data_path, train=True, test=False, rate1=1)
    # vaL_dataset =antibody_antigen_dataset(antigen_config=antigen_config,antibody_config=antibody_config,data_path=data path, train=False, test=True, rate1=0.7)

    if not args.skip_precompute:
        train_dataset.precompute_embeddings()
    train_dataset.release_encoders()

    train_dataloader = DataLoader(train_dataset, shuffle=False, batch_size=args.batch_size)
    # vaL_dataloader = DataLoader(val_dataset, shuffLe=False, batch_size=args.batch_size)


    model = antibinder(antibody_hidden_dim=1024,antigen_hidden_dim=1024,latent_dim=args.latent_dim,res=False).cuda()
    print(model)

    # Multi-GPU: wrap the WHOLE model (per-submodule wrapping breaks because
    # forward passes nested lists between the submodules).
    n_gpus = torch.cuda.device_count()
    print(f"Visible GPUs: {n_gpus}")
    if n_gpus > 1:
        model = nn.DataParallel(model)
        print(f"Training with DataParallel on {n_gpus} GPUs.")

    os.makedirs('./logs', exist_ok=True)
    os.makedirs('./ckpts', exist_ok=True)
    logger = CSVLogger_my(['epoch', 'train_loss', 'train_acc', 'train_precision', 'train_f1', 'train_recall'],f"./logs/{args.model_name}_{args.data}_{args.batch_size}_{args.epochs}_{args.latent_dim}_{args.lr}.csv")
    scheduler = None

    # load model if needs
    load = False
    if load:
        weight = torch.load('')
        raw_model = model.module if isinstance(model, nn.DataParallel) else model
        raw_model.load_state_dict(weight)
        print("load model success")


    trainer = Trainer(model=model,
        train_dataloader=train_dataloader,
        # valid_dataloader=val_dataLoader,
        # test_dataLoader=test._dataloader,
        logger = logger,
        args= args,
        load=load
        )

    criterion = nn.BCELoss()
    trainer.train(criterion=criterion, epochs=args.epochs)