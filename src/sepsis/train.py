"""
Training step (runs locally or as a SageMaker Training job).

Everything that involves a choice (best epoch, alert thresholds) is decided
on the validation split, using the challenge's utility score. The test split is not read here at all; it is
scored once by sepsis.evaluate.

Usage:
  python -m sepsis.train --data-dir data/processed --model-dir models/production
"""

import argparse
import copy
import os
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

from . import metrics
from .data import HourlySplit
from .features import WINDOW_SIZE, Preprocessor
from .labels import HORIZON
from .model import PREPROCESSOR_FILE, SepsisLSTM, save_bundle
from .utility import UtilityScorer


def parse_args():
    # Defaults follow SageMaker Training's environment variables
    p = argparse.ArgumentParser()
    p.add_argument('--data-dir',  default=os.environ.get('SM_CHANNEL_TRAIN', 'data/processed'))
    p.add_argument('--model-dir', default=os.environ.get('SM_MODEL_DIR', 'models/production'))
    p.add_argument('--hidden-size',  type=int,   default=64)
    p.add_argument('--num-layers',   type=int,   default=2)
    p.add_argument('--dropout',      type=float, default=0.3)
    p.add_argument('--lr',           type=float, default=1e-3)
    p.add_argument('--weight-decay', type=float, default=1e-5)
    p.add_argument('--batch-size',   type=int,   default=1024)
    p.add_argument('--epochs',       type=int,   default=30)
    p.add_argument('--patience',     type=int,   default=5)
    p.add_argument('--pos-weight',   type=float, default=1.0,
                   help='>1 up-weights positives. 1.0 keeps probabilities calibrated; '
                        'the alert thresholds already handle the class imbalance.')
    p.add_argument('--seed',         type=int,   default=42)
    return p.parse_args()


def set_seed(seed: int) -> None:
    np.random.seed(seed)
    torch.manual_seed(seed)


def train_epoch(model, X, y, optimizer, criterion, batch_size) -> float:
    model.train()
    order = torch.randperm(len(X), device=X.device)
    total = 0.0
    for i in range(0, len(X), batch_size):
        idx = order[i:i + batch_size]
        optimizer.zero_grad()
        loss = criterion(model(X[idx]), y[idx])
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        total += loss.item() * len(idx)
    return total / len(X)


def main():
    args = parse_args()
    set_seed(args.seed)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    data_dir = Path(args.data_dir)

    preprocessor = Preprocessor.from_json(data_dir / PREPROCESSOR_FILE)
    train = HourlySplit.load(data_dir / 'train.npz')
    val   = HourlySplit.load(data_dir / 'val.npz')
    X_tr, y_tr, _     = train.windows(WINDOW_SIZE)
    X_va, y_va, p_va  = val.windows(WINDOW_SIZE)
    print(f'device={device} train_windows={len(X_tr)} (pos {y_tr.mean():.2%}) '
          f'val_windows={len(X_va)} (pos {y_va.mean():.2%})')

    X_tr_t = torch.from_numpy(X_tr).to(device)
    y_tr_t = torch.from_numpy(y_tr).to(device)

    model = SepsisLSTM(X_tr.shape[2], args.hidden_size,
                       args.num_layers, args.dropout).to(device)
    criterion = nn.BCEWithLogitsLoss(
        pos_weight=torch.tensor(args.pos_weight, device=device))
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr,
                                 weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='max', factor=0.5, patience=2)

    val_scorer = UtilityScorer(y_va, p_va)

    # Model selection uses the challenge utility (at its best threshold);
    # AUROC is logged alongside for reference
    history, best_util, best_state, best_epoch, stale = [], -np.inf, None, 0, 0
    for epoch in range(1, args.epochs + 1):
        start = time.time()
        loss = train_epoch(model, X_tr_t, y_tr_t, optimizer, criterion,
                           args.batch_size)
        probs = model.predict_proba(X_va)
        val_auroc = metrics.summary(y_va, probs)['auroc']
        _, val_util = val_scorer.best_threshold(probs)
        scheduler.step(val_util)
        history.append({'epoch': epoch, 'train_loss': loss,
                        'val_auroc': val_auroc, 'val_utility': val_util})

        improved = val_util > best_util
        if improved:
            best_util, best_epoch, stale = val_util, epoch, 0
            best_state = copy.deepcopy(model.state_dict())
        else:
            stale += 1
        # key=value format so SageMaker metric_definitions can scrape it
        print(f'epoch={epoch} train_loss={loss:.4f} val_auroc={val_auroc:.4f} '
              f'val_utility={val_util:.4f} '
              f'lr={optimizer.param_groups[0]["lr"]:.1e} '
              f'time={time.time() - start:.0f}s' + (' *' if improved else ''))
        if stale >= args.patience:
            print(f'Early stopping: no improvement for {args.patience} epochs')
            break

    model.load_state_dict(best_state)
    model.to('cpu')
    p_val = model.predict_proba(X_va)

    decision, val_util = val_scorer.best_threshold(p_val)
    thresholds = metrics.alert_thresholds(y_va, p_val, decision)
    val_report = {
        **metrics.summary(y_va, p_val),
        'utility':      val_util,
        'at_threshold': metrics.at_threshold(y_va, p_val, decision),
        'patient_level': metrics.patient_level(y_va, p_val, p_va, decision,
                                               val_scorer.hours_to_sepsis),
    }
    print(f'best_epoch={best_epoch} val_utility={val_util:.4f} '
          f'val_auroc={val_report["auroc"]:.4f} decision_threshold={decision:.4f}')

    config = {
        'model_version': datetime.now(timezone.utc).strftime('%Y%m%d-%H%M%S'),
        'feature_cols':  list(preprocessor.feature_cols),
        'window_size':   WINDOW_SIZE,
        'horizon_hours': HORIZON,
        'architecture':  {'hidden_size': args.hidden_size,
                          'num_layers':  args.num_layers,
                          'dropout':     args.dropout},
        'training':      {**{k: v for k, v in vars(args).items()
                             if k not in ('data_dir', 'model_dir')},
                          'best_epoch': best_epoch, 'history': history},
        'decision_threshold': decision,
        'alert_thresholds':   thresholds,
        'validation':         val_report,
    }

    rng = np.random.default_rng(args.seed)
    background = X_va[rng.choice(len(X_va), size=min(100, len(X_va)), replace=False)]

    save_bundle(args.model_dir, model, config, preprocessor, background)
    print(f'Saved model bundle -> {args.model_dir}')


if __name__ == '__main__':
    main()
