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
sys.path.append('../') 
import warnings 
warnings.filterwarnings("ignore")


class Trainer():
    def __init__(self, model, train_dataloader, args, logger, valid_dataloader=None, load=False) -> None:
        self.model = model
        self.train_dataloader = train_dataloader
        self.valid_dataloader = valid_dataloader
        self.args = args
        self.logger = logger
        self.best_loss = None
        # Checkpoints are selected on validation F1 when a val split exists.
        self.best_val_f1 = None
        self.epochs_no_improve = 0
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

    @torch.no_grad()
    def validate(self, criterion):
        """Evaluate on the held-out validation split. No grads / optimizer."""
        self.model.eval()
        val_loss = 0.0
        num_val = 0
        Y_hat, Y = [], []
        for antibody_set, antigen_set, label in tqdm(self.valid_dataloader, desc='Validating'):
            with torch.cuda.amp.autocast():
                probs = self.model(antibody_set, antigen_set)
            probs = probs.float()
            y = label.float().cuda()
            # Sum of per-sample BCE (reduction='mean' averaged over the batch;
            # multiply by batch size, then divide by the total at the end so
            # the partial last batch is weighted correctly).
            val_loss += criterion(probs.view(-1), y.view(-1)).item() * y.shape[0]
            num_val += y.shape[0]
            Y_hat.append((probs > 0.5).long().reshape(-1))
            Y.append(y.reshape(-1))

        val_acc, val_precision, val_f1, val_recall = self.matrix_val(
            torch.cat(Y_hat).long().cpu().numpy(),
            torch.cat(Y).cpu().numpy())
        val_loss = np.exp(val_loss / num_val)
        return val_loss, val_acc, val_precision, val_f1, val_recall

    def train(self, criterion, epochs):
        optimizer = torch.optim.Adam(self.model.parameters(), lr=self.args.lr)
        # Mixed precision: roughly halves activation memory and speeds up T4s.
        scaler = torch.cuda.amp.GradScaler()
        accum = max(1, self.args.grad_accum)
        for epoch in range(epochs):
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

                # BCELoss is hard-blocked inside autocast and numerically
                # unstable in fp16, so cast sigmoid outputs to fp32 and
                # compute the loss OUTSIDE the autocast region.
                probs = probs.float()
                y = label.float().cuda()
                loss = criterion(probs.view(-1), y.view(-1)) / accum

                yhat = (probs > 0.5).long()
                # Backprop still reaches the fp16 graph; GradScaler handles
                # the mixed-precision gradient scaling.
                scaler.scale(loss).backward()

                # Flatten per batch: the final batch can be shorter (17 vs 32),
                # so rows can't be stacked into a rectangular array.
                Y_hat.append(yhat.reshape(-1))
                Y.append(y.reshape(-1))

                if (step + 1) % accum == 0 or (step + 1) == len(self.train_dataloader):
                    scaler.step(optimizer)
                    scaler.update()
                    optimizer.zero_grad()

                train_loss += loss.item() * accum
                num_train += antibody_set[0].shape[0]

            # Move collected GPU tensors to CPU before sklearn metrics.
            train_acc, train_precision, train_f1, train_recall = self.matrix_val(
                torch.cat(Y_hat).long().cpu().numpy(),
                torch.cat(Y).cpu().numpy())
            train_loss = train_loss / num_train
            train_loss = np.exp(train_loss)

            print(f"Epoch {epoch+1} | train_loss: {train_loss:.4f}, train_acc: {train_acc:.4f}, "
                  f"train_precision: {train_precision:.4f}, train_f1: {train_f1:.4f}, train_recall: {train_recall:.4f}")

            if self.valid_dataloader is not None:
                val_loss, val_acc, val_precision, val_f1, val_recall = self.validate(criterion)
                print(f"Epoch {epoch+1} | val_loss: {val_loss:.4f}, val_acc: {val_acc:.4f}, "
                      f"val_precision: {val_precision:.4f}, val_f1: {val_f1:.4f}, val_recall: {val_recall:.4f}")
                self.logger.log([epoch+1, train_loss, train_acc, train_precision, train_f1, train_recall,
                                 val_loss, val_acc, val_precision, val_f1, val_recall])

                # Select checkpoints on held-out validation F1.
                if self.best_val_f1 is None or val_f1 > self.best_val_f1:
                    print(f"val_f1 improved ({self.best_val_f1} -> {val_f1:.4f}), saving...")
                    self.best_val_f1 = val_f1
                    self.epochs_no_improve = 0
                    self.save_model()
                else:
                    self.epochs_no_improve += 1
                    print(f"val_f1 did not improve for {self.epochs_no_improve} epoch(s) "
                          f"(best: {self.best_val_f1:.4f}).")
                    if self.args.patience > 0 and self.epochs_no_improve >= self.args.patience:
                        print(f"Early stopping at epoch {epoch+1}. Best val_f1: {self.best_val_f1:.4f}")
                        break
            else:
                # No val split: fall back to training-loss selection.
                self.logger.log([epoch+1, train_loss, train_acc, train_precision, train_f1, train_recall])
                if self.best_loss is None or train_loss < self.best_loss:
                    print('epoch: ', epoch, ' saving...')
                    self.best_loss = train_loss
                    self.save_model()

    def save_model(self):
        # DataParallel wraps the model in .module; save unwrapped weights
        # so checkpoints load directly into antibinder(...) later.
        raw_model = self.model.module if isinstance(self.model, nn.DataParallel) else self.model
        torch.save(raw_model.state_dict(), f"./ckpts/{self.args.model_name}_{self.args.data}_{self.args.batch_size}_{self.args.epochs}_{self.args.latent_dim}_{self.args.lr}.pth")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--seed', type=int, default=42)
    # Global batch size (split across GPUs by DataParallel). 64 = 32 per T4.
    parser.add_argument('--batch_size', type=int, default=64)
    # Effective batch = batch_size * grad_accum (64 * 2 = 128 by default).
    parser.add_argument('--grad_accum', type=int, default=2)
    parser.add_argument('--latent_dim', type=int, default=36)
    # Fraction of rows used for TRAINING; the rest is the held-out validation
    # set. Set to 1.0 to disable validation (checkpoint falls back to train loss).
    parser.add_argument('--train_rate', type=float, default=0.8)
    # Stop after this many consecutive epochs without val-F1 improvement.
    # 0 disables early stopping.
    parser.add_argument('--patience', type=int, default=30)
    # In certain datasets, an early stopping strategy is required to achieve optimal results.
    parser.add_argument('--epochs', type=int, default=500)
    # parser.add_argument('--weight_decay', type=float, default=1e-5, help='weight decay used in optimizer') # 1e-5
    parser.add_argument('--lr', type=float, default=6e-5, help='learning rate')
    parser.add_argument('--model_name', type=str, default='AntiBinder')
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

    # Shuffle rows BEFORE the positional 80/20 split so both splits cover all
    # antigens/classes (the CSV rows are not randomly ordered). Both datasets
    # receive the same shuffled frame, so their slices are complementary.
    df = pd.read_csv(data_path)
    df = df.dropna(subset=['H-FR1','H-CDR1','H-FR2','H-CDR2','H-FR3','H-CDR3','H-FR4'])
    df = df.sample(frac=1.0, random_state=args.seed).reset_index(drop=True)

    # NOTE: the dataset constructor loads ESM-2 650M and 4 IgFold models onto
    # the GPU. Build the datasets FIRST, precompute all embeddings, and release
    # those encoders before allocating the training model.
    train_dataset = antibody_antigen_dataset(antigen_config=antigen_config, antibody_config=antibody_config,
                                              data=df, train=True, test=False, rate1=args.train_rate)
    val_dataset = None
    if args.train_rate < 1.0:
        val_dataset = antibody_antigen_dataset(antigen_config=antigen_config, antibody_config=antibody_config,
                                                data=df, train=False, test=True, rate1=args.train_rate,
                                                share_encoders=train_dataset)

    print("Precomputing embeddings for train split...")
    train_dataset.precompute_embeddings()
    train_dataset.release_encoders()
    if val_dataset is not None:
        print("Precomputing embeddings for validation split...")
        val_dataset.precompute_embeddings()
        val_dataset.release_encoders()

    train_dataloader = DataLoader(train_dataset, shuffle=True, batch_size=args.batch_size)
    val_dataloader = None
    if val_dataset is not None:
        val_dataloader = DataLoader(val_dataset, shuffle=False, batch_size=args.batch_size)


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
    log_columns = ['epoch', 'train_loss', 'train_acc', 'train_precision', 'train_f1', 'train_recall']
    if val_dataloader is not None:
        log_columns += ['val_loss', 'val_acc', 'val_precision', 'val_f1', 'val_recall']
    logger = CSVLogger_my(log_columns, f"./logs/{args.model_name}_{args.data}_{args.batch_size}_{args.epochs}_{args.latent_dim}_{args.lr}.csv")
    scheduler = None

    # load model if needs
    load = False
    if load:
        weight = torch.load('')
        raw_model = model.module if isinstance(model, nn.DataParallel) else model
        raw_model.load_state_dict(weight)
        print("load model success")


    trainer = Trainer(
        model=model,
        train_dataloader=train_dataloader,
        valid_dataloader=val_dataloader,
        logger=logger,
        args=args,
        load=load
    )

    criterion = nn.BCELoss()
    trainer.train(criterion=criterion, epochs=args.epochs)