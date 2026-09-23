import os
# Kaggle T4 x2: expose both GPUs so DataParallel can split each batch
# (embedding precomputation with ESM/IgFold still runs on cuda:0).
os.environ["CUDA_VISIBLE_DEVICES"] = '0,1'

from antigen_antibody_emb import *
from antibinder_model import *
import torch
import torch.nn as nn
import numpy as np
import pandas as pd
import argparse
import sys
import warnings
from torch.utils.data import DataLoader
from tqdm import tqdm
from utils.utils import CSVLogger_my
from sklearn.metrics import (accuracy_score, precision_score, f1_score, recall_score,
                             roc_auc_score, confusion_matrix)

sys.path.append('../')
warnings.filterwarnings("ignore")


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------
def classification_metrics(y, yhat):
    """Train/val metrics from hard 0/1 predictions."""
    return (accuracy_score(y, yhat),
            precision_score(y, yhat),
            f1_score(y, yhat),
            recall_score(y, yhat))


def test_metrics(y, yhat, yscores):
    """Test metrics: ROC-AUC + confusion-matrix counts. Returns
    (auc, precision, accuracy, recall, f1, TN, FP, FN, TP)."""
    cm = confusion_matrix(y, yhat, labels=[0, 1]).ravel()
    TN, FP, FN, TP = cm
    # AUC is undefined when only one class is present.
    auc = roc_auc_score(y, yscores) if len(np.unique(y)) > 1 else None
    return (auc,
            precision_score(y, yhat),
            accuracy_score(y, yhat),
            recall_score(y, yhat),
            f1_score(y, yhat),
            int(TN), int(FP), int(FN), int(TP))


# ---------------------------------------------------------------------------
# Trainer: train loop, held-out validation, and final testing
# ---------------------------------------------------------------------------
class Trainer():
    def __init__(self, model, args, logger, train_dataloader=None,
                 valid_dataloader=None, load=False,
                 best_loss=None, best_val_f1=None) -> None:
        self.model = model
        self.args = args
        self.logger = logger
        self.train_dataloader = train_dataloader
        self.valid_dataloader = valid_dataloader
        self.best_loss = best_loss
        # Checkpoints are selected on validation F1 when a val split exists.
        self.best_val_f1 = best_val_f1
        self.epochs_no_improve = 0
        self.load = load

        if not self.load:
            self.init()
        else:
            print("no init model")

    def init(self):
        init = AntiModelIinitial()
        self.model.apply(init._init_weights)
        print("init successfully!")

    # ----- checkpoint paths ------------------------------------------------
    @property
    def ckpt_path(self):
        # Stable, metric-independent pointer to the best weights. Auto-load
        # at startup / test time finds this file purely by name.
        return f"./ckpts/{self.args.model_name}_best.pth"

    def save_model(self, metric_value, metric_name):
        # DataParallel wraps the model in .module; save unwrapped weights
        # so checkpoints load directly into antibinder(...) later.
        raw_model = self.model.module if isinstance(self.model, nn.DataParallel) else self.model
        # Store weights together with the metric they achieved, so a resumed
        # run doesn't overwrite the best checkpoint with an inferior first epoch.
        payload = {
            'state_dict': raw_model.state_dict(),
            'metric_name': metric_name,
            metric_name: float(metric_value),
        }
        # History file carrying the metric value, e.g. AntiBinder_valf1_0.6267.pth
        metric_path = f"./ckpts/{self.args.model_name}_{metric_name}_{metric_value:.4f}.pth"
        torch.save(payload, metric_path)
        # Overwrite the stable "best" pointer with the same payload.
        torch.save(payload, self.ckpt_path)
        print(f"checkpoint saved: {metric_path} (also {self.ckpt_path})")

    # ----- validation ------------------------------------------------------
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
            probs = probs.float().view(-1)
            y = label.float().cuda().view(-1)
            # Batch mean * batch size, divided by the total at the end so the
            # partial last batch is weighted correctly.
            val_loss += criterion(probs, y).item() * y.shape[0]
            num_val += y.shape[0]
            Y_hat.append((probs > 0.5).long().cpu())
            Y.append(y.cpu())

        val_acc, val_precision, val_f1, val_recall = classification_metrics(
            torch.cat(Y_hat).numpy(), torch.cat(Y).numpy())
        # Plain mean binary cross-entropy (0.0 best, ~0.693 = random).
        val_loss = val_loss / num_val
        return val_loss, val_acc, val_precision, val_f1, val_recall

    # ----- training --------------------------------------------------------
    def train(self, criterion, epochs):
        optimizer = torch.optim.Adam(self.model.parameters(), lr=self.args.lr)
        # Mixed precision: roughly halves activation memory and speeds up T4s.
        scaler = torch.cuda.amp.GradScaler()
        accum = max(1, self.args.grad_accum)
        for epoch in range(epochs):
            self.model.train(True)
            train_loss = 0.0
            num_train = 0
            Y_hat, Y = [], []
            optimizer.zero_grad()
            for step, (antibody_set, antigen_set, label) in enumerate(tqdm(self.train_dataloader)):
                with torch.cuda.amp.autocast():
                    probs = self.model(antibody_set, antigen_set)

                # BCELoss is hard-blocked inside autocast and numerically
                # unstable in fp16, so cast sigmoid outputs to fp32 and
                # compute the loss OUTSIDE the autocast region.
                probs = probs.float().view(-1)
                y = label.float().cuda().view(-1)
                loss = criterion(probs, y) / accum

                Y_hat.append((probs > 0.5).long().cpu())
                Y.append(y.cpu())
                # Backprop still reaches the fp16 graph; GradScaler handles
                # the mixed-precision gradient scaling.
                scaler.scale(loss).backward()

                if (step + 1) % accum == 0 or (step + 1) == len(self.train_dataloader):
                    scaler.step(optimizer)
                    scaler.update()
                    optimizer.zero_grad()

                train_loss += loss.item() * accum
                num_train += y.shape[0]

            train_acc, train_precision, train_f1, train_recall = classification_metrics(
                torch.cat(Y_hat).numpy(), torch.cat(Y).numpy())
            # Plain mean binary cross-entropy (0.0 best, ~0.693 = random).
            train_loss = train_loss / num_train

            print(f"Epoch {epoch+1} | train_loss: {train_loss:.4f}, train_acc: {train_acc:.4f}, "
                  f"train_precision: {train_precision:.4f}, train_f1: {train_f1:.4f}, "
                  f"train_recall: {train_recall:.4f}")

            if self.valid_dataloader is not None:
                val_loss, val_acc, val_precision, val_f1, val_recall = self.validate(criterion)
                print(f"Epoch {epoch+1} | val_loss: {val_loss:.4f}, val_acc: {val_acc:.4f}, "
                      f"val_precision: {val_precision:.4f}, val_f1: {val_f1:.4f}, "
                      f"val_recall: {val_recall:.4f}")
                self.logger.log([epoch+1, train_loss, train_acc, train_precision, train_f1, train_recall,
                                 val_loss, val_acc, val_precision, val_f1, val_recall])

                # Select checkpoints on held-out validation F1.
                if self.best_val_f1 is None or val_f1 > self.best_val_f1:
                    print(f"val_f1 improved ({self.best_val_f1} -> {val_f1:.4f}), saving...")
                    self.best_val_f1 = val_f1
                    self.epochs_no_improve = 0
                    self.save_model(val_f1, 'valf1')
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
                    self.save_model(train_loss, 'trainloss')

    # ----- testing ---------------------------------------------------------
    @torch.no_grad()
    def test(self):
        """Final evaluation: AUC, hard-metric scores and the confusion matrix."""
        self.model.eval()
        Y, Y_hat, Y_scores = [], [], []
        for antibody_set, antigen_set, label in tqdm(self.valid_dataloader, desc='Testing'):
            with torch.cuda.amp.autocast():
                probs = self.model(antibody_set, antigen_set)
            probs = probs.float().cpu().view(-1)
            y = label.float().view(-1)
            Y.append(y)
            Y_hat.append((probs > 0.5).long())
            Y_scores.append(probs)

        metrics = test_metrics(torch.cat(Y).numpy(),
                               torch.cat(Y_hat).numpy(),
                               torch.cat(Y_scores).numpy())
        auc, precision, acc, recall, f1, TN, FP, FN, TP = metrics
        print(f"\nTest | AUC: {auc}, Precision: {precision:.4f}, Acc: {acc:.4f}, "
              f"Recall: {recall:.4f}, F1: {f1:.4f}")
        print(f"Confusion matrix | TN: {TN}, FP: {FP}, FN: {FN}, TP: {TP}")
        self.logger.log([auc, precision, acc, recall, f1, TN, FP, FN, TP])
        return metrics


# ---------------------------------------------------------------------------
# Shared construction helpers
# ---------------------------------------------------------------------------
def build_configs():
    antigen_config = configuration()
    setattr(antigen_config, 'max_position_embeddings', 1024)
    antibody_config = configuration()
    setattr(antibody_config, 'max_position_embeddings', 149)
    return antigen_config, antibody_config


def ckpt_path_for(args):
    return f"./ckpts/{args.model_name}_best.pth"


def extract_state_dict(ckpt):
    # New format wraps weights in {'state_dict':...}; legacy files are raw state_dicts.
    return ckpt['state_dict'] if isinstance(ckpt, dict) and 'state_dict' in ckpt else ckpt


def build_model(args, require_ckpt=False):
    """Create the model, optionally load the best checkpoint, then wrap for
    multi-GPU. Returns (model, load_flag, resume_val_f1, resume_loss)."""
    model = antibinder(antibody_hidden_dim=1024, antigen_hidden_dim=1024,
                       latent_dim=args.latent_dim, res=False).cuda()

    ckpt_path = ckpt_path_for(args)
    load = os.path.exists(ckpt_path) and not getattr(args, 'fresh', False)
    resume_val_f1, resume_loss = None, None
    if load:
        ckpt = torch.load(ckpt_path, map_location='cuda:0')
        model.load_state_dict(extract_state_dict(ckpt))
        if isinstance(ckpt, dict):
            resume_val_f1 = ckpt.get('valf1')
            resume_loss = ckpt.get('trainloss')
        print(f"Resumed weights from {ckpt_path} "
              f"(val_f1={resume_val_f1}, train_loss={resume_loss})")
    else:
        if require_ckpt:
            raise FileNotFoundError(
                f"No checkpoint at {ckpt_path}. Train first (python main.py train) "
                f"or check --model_name."
            )
        print(f"No checkpoint found at {ckpt_path} (or --fresh given); training from scratch.")

    n_gpus = torch.cuda.device_count()
    print(f"Visible GPUs: {n_gpus}")
    if n_gpus > 1:
        model = nn.DataParallel(model)
        print(f"Using DataParallel on {n_gpus} GPUs.")
    return model, load, resume_val_f1, resume_loss


# ---------------------------------------------------------------------------
# Subcommand: train
# ---------------------------------------------------------------------------
def run_train(args):
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    antigen_config, antibody_config = build_configs()

    if not os.path.exists(args.data_path):
        raise FileNotFoundError(
            f"Data CSV not found: {args.data_path}. "
            f"Pass a valid file with --data_path (cwd={os.getcwd()})"
        )

    # Two supported CSV layouts:
    #  * prepared files with a `split` column (SEPIQ): cluster-disjoint
    #    train/val is taken verbatim from the file, --train_rate is ignored.
    #  * legacy single-split CSVs (COVID/HIV): rows are shuffled (seeded) and
    #    split positionally by --train_rate.
    df = pd.read_csv(args.data_path)
    df = df.dropna(subset=['H-FR1', 'H-CDR1', 'H-FR2', 'H-CDR2', 'H-FR3', 'H-CDR3', 'H-FR4'])

    # NOTE: the dataset constructor loads ESM-2 650M and 4 IgFold models onto
    # the GPU. Build the datasets FIRST, precompute all embeddings, and release
    # those encoders before allocating the training model.
    if 'split' in df.columns:
        train_df = df[df['split'] == 'train'].sample(frac=1.0, random_state=args.seed).reset_index(drop=True)
        val_df = df[df['split'] == 'val'].reset_index(drop=True)
        if len(train_df) == 0 or len(val_df) == 0:
            raise ValueError(f"split column in {args.data_path} must contain both "
                             f"'train' and 'val' rows (got {len(train_df)}/{len(val_df)}).")
        print(f"Using file-provided cluster-disjoint split: train={len(train_df)}, val={len(val_df)} "
              f"(positional --train_rate={args.train_rate} ignored).")
        train_dataset = antibody_antigen_dataset(antigen_config=antigen_config, antibody_config=antibody_config,
                                                  data=train_df, train=True, test=False, rate1=1.0)
        val_dataset = antibody_antigen_dataset(antigen_config=antigen_config, antibody_config=antibody_config,
                                                data=val_df, train=False, test=True, rate1=0.0,
                                                share_encoders=train_dataset)
    else:
        # Shuffle rows BEFORE the positional train/val split so both splits
        # cover all antigens/classes (the CSV rows are not randomly ordered).
        df = df.sample(frac=1.0, random_state=args.seed).reset_index(drop=True)
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

    model, load, resume_val_f1, resume_loss = build_model(args)

    os.makedirs('./logs', exist_ok=True)
    os.makedirs('./ckpts', exist_ok=True)
    log_columns = ['epoch', 'train_loss', 'train_acc', 'train_precision', 'train_f1', 'train_recall']
    if val_dataloader is not None:
        log_columns += ['val_loss', 'val_acc', 'val_precision', 'val_f1', 'val_recall']
    logger = CSVLogger_my(
        log_columns,
        f"./logs/{args.model_name}_{args.data}_{args.batch_size}_{args.epochs}_{args.latent_dim}_{args.lr}.csv")

    trainer = Trainer(model=model, args=args, logger=logger,
                      train_dataloader=train_dataloader,
                      valid_dataloader=val_dataloader,
                      load=load,
                      best_loss=resume_loss,
                      best_val_f1=resume_val_f1)

    criterion = nn.BCELoss()
    trainer.train(criterion=criterion, epochs=args.epochs)


# ---------------------------------------------------------------------------
# Subcommand: test
# ---------------------------------------------------------------------------
def run_test(args):
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    antigen_config, antibody_config = build_configs()
    print("antigen_config:\n", antigen_config)
    print("antibody_config:\n", antibody_config)

    if not os.path.exists(args.data_path):
        raise FileNotFoundError(
            f"Data CSV not found: {args.data_path}. "
            f"Pass a valid file with --data_path (cwd={os.getcwd()})"
        )
    print(args.data_path)

    # Prepared SEPIQ-style CSVs carry a cluster-disjoint `split` column; by
    # default the test command evaluates the held-out 'val' rows only.
    # --eval_split all keeps the legacy behavior (evaluate the WHOLE csv).
    eval_df = pd.read_csv(args.data_path)
    if 'split' in eval_df.columns and args.eval_split != 'all':
        eval_df = eval_df[eval_df['split'] == args.eval_split].reset_index(drop=True)
        if len(eval_df) == 0:
            raise ValueError(f"No rows with split == '{args.eval_split}' in {args.data_path}.")
        print(f"Evaluating split='{args.eval_split}': {len(eval_df)} rows.")
        kwargs = {'data': eval_df}
    else:
        kwargs = {'data_path': args.data_path}

    # rate1=0 with test=True selects the WHOLE csv/frame (iloc[0:]).
    test_dataset = antibody_antigen_dataset(antigen_config=antigen_config, antibody_config=antibody_config,
                                            train=False, test=True, rate1=0, **kwargs)
    print("Precomputing embeddings for test split...")
    test_dataset.precompute_embeddings()
    test_dataset.release_encoders()
    test_dataloader = DataLoader(test_dataset, shuffle=False, batch_size=args.batch_size)

    # Checkpoint is mandatory for testing.
    model, _, _, _ = build_model(args, require_ckpt=True)

    os.makedirs('./logs', exist_ok=True)
    logger = CSVLogger_my(
        ['test_auc', 'test_precision', 'test_acc', 'test_recall', 'test_f1', 'TN', 'FP', 'FN', 'TP'],
        f"./logs/{args.model_name}_{args.latent_dim}_{args.data}.csv")

    trainer = Trainer(model=model, args=args, logger=logger, valid_dataloader=test_dataloader, load=True)
    trainer.test()
    print("\n\nSuccess!!!")


# ---------------------------------------------------------------------------
# CLI: python main.py {train,test} [options]
# ---------------------------------------------------------------------------
def build_parser():
    # Preserve newlines in description/epilog examples (RawDescription) while
    # still appending "(default: ...)" to every argument help (Defaults).
    class HelpFormatter(argparse.RawDescriptionHelpFormatter,
                        argparse.ArgumentDefaultsHelpFormatter):
        pass
    fmt = HelpFormatter

    # Arguments shared by both subcommands.
    common = argparse.ArgumentParser(add_help=False)
    common.add_argument('--seed', type=int, default=42,
                        help='random seed for PyTorch, NumPy and the train/val row shuffle')
    common.add_argument('--batch_size', type=int, default=64,
                        help='global batch size, split across all visible GPUs by DataParallel '
                             '(64 = 32 per T4)')
    common.add_argument('--latent_dim', type=int, default=36,
                        help='latent dimension after the cross-attention projection')
    common.add_argument('--model_name', type=str, default='AntiBinder',
                        help='name prefix for checkpoints (ckpts/<name>_best.pth) and logs')
    common.add_argument('--data_path', type=str,
                        default='./datasets/process_data/SEPIQ/AVIDa-hHER2_binary.csv',
                        help='path to the prepared CSV; must contain H-FR1..H-FR4, vh, "Antigen", '
                             '"Antigen Sequence" and ANT_Binding columns. A "split" column '
                             '(train/val, e.g. from prepare_sepiq.py) enables cluster-disjoint '
                             'train/val handling')

    parser = argparse.ArgumentParser(
        prog='python main.py',
        formatter_class=fmt,
        description='AntiBinder: antibody-antigen binding prediction (train / test entry point)')
    sub = parser.add_subparsers(dest='command', required=True, metavar='{train,test}')

    # ---- train ----
    p_train = sub.add_parser(
        'train', parents=[common], formatter_class=fmt,
        help='train the model with a held-out validation split and early stopping',
        description='Train AntiBinder. Rows are shuffled (seeded) and split into train/val; '
                    'ESM-2/IgFold embeddings are precomputed once and cached, then the encoder '
                    'models are released before AntiBinder training starts. The best checkpoint '
                    'is selected on validation F1 and saved to ckpts/<model_name>_best.pth. '
                    'Training auto-resumes from that file unless --fresh is given.',
        epilog='examples:\n'
               '  # SEPIQ AVIDa-hHER2 (run prepare_sepiq.py first)\n'
               '  python main.py train --data_path ./datasets/process_data/SEPIQ/AVIDa-hHER2_binary.csv\n'
               '  python main.py train --epochs 100 --batch_size 64 --lr 6e-5\n'
               '  # legacy CSV without a split column\n'
               '  python main.py train --train_rate 1.0          # no val split, select on train loss\n'
               '  python main.py train --fresh                   # ignore existing checkpoint')
    p_train.add_argument('--data', type=str, default='train',
                         help='label used in the training log file name')
    # Effective batch = batch_size * grad_accum (64 * 2 = 128 by default).
    p_train.add_argument('--grad_accum', type=int, default=2,
                         help='gradient accumulation steps; effective batch = batch_size x grad_accum '
                              '(64 x 2 = 128)')
    # Fraction of rows for TRAINING; the rest is the held-out validation set.
    # 1.0 disables validation (checkpoint selection falls back to train loss).
    p_train.add_argument('--train_rate', type=float, default=0.8,
                         help='fraction of rows used for training; the remainder is validation. '
                              'Set 1.0 to disable validation')
    # Stop after this many epochs without val-F1 improvement. 0 disables it.
    p_train.add_argument('--patience', type=int, default=30,
                         help='stop after N consecutive epochs without val-F1 improvement; '
                              '0 disables early stopping')
    p_train.add_argument('--epochs', type=int, default=500,
                         help='maximum number of training epochs')
    p_train.add_argument('--lr', type=float, default=6e-5, help='Adam learning rate')
    # Ignore an existing ckpts/<model_name>_best.pth and reinitialize weights.
    p_train.add_argument('--fresh', action='store_true',
                         help='ignore an existing best checkpoint and reinitialize all weights')
    p_train.set_defaults(func=run_train)

    # ---- test ----
    p_test = sub.add_parser(
        'test', parents=[common], formatter_class=fmt,
        help='evaluate the best checkpoint on a CSV (val split by default)',
        description='Evaluate ckpts/<model_name>_best.pth on the CSV given by --data_path. '
                    'When the CSV has a "split" column (prepare_sepiq.py output), only the '
                    "held-out 'val' rows are evaluated by default (--eval_split val); use "
                    "--eval_split all for the ENTIRE CSV. Reports ROC-AUC, accuracy, precision, "
                    'recall, F1 and the TN/FP/FN/TP confusion matrix, and appends one row to '
                    'logs/<model_name>_<latent_dim>_test.csv. Embeddings are precomputed/cached '
                    'first as in training.',
        epilog='examples:\n'
               '  python main.py test\n'
               '  python main.py test --eval_split all\n'
               '  python main.py test --data_path ./datasets/process_data/HIV/dataset_hiv_split.csv')
    p_test.add_argument('--data', type=str, default='test',
                        help='label used in the test log file name')
    p_test.add_argument('--eval_split', choices=['val', 'train', 'all'], default='val',
                        help="which rows to evaluate when the CSV contains a 'split' column; "
                             "'all' ignores the column (legacy behavior)")
    p_test.set_defaults(func=run_test)

    return parser


if __name__ == "__main__":
    args = build_parser().parse_args()
    args.func(args)
