import torch
import random
import numpy as np
from mmengine import fileio
import io
import os
import json

def openjson(path):
       value  = fileio.get_text(path)
       dict = json.loads(value)
       return dict

def opendata(path):
    
    npz_bytes = fileio.get(path)
    buff = io.BytesIO(npz_bytes)
    npz_data = np.load(buff, allow_pickle=True)

    return npz_data

def set_seed(CUR_SEED):
    random.seed(CUR_SEED)
    np.random.seed(CUR_SEED)
    torch.manual_seed(CUR_SEED)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def get_epoch_mean_loss(epoch_loss):
    epoch_mean_loss = {}
    for current_loss in epoch_loss:
        for key, value in current_loss.items():
            if key in epoch_mean_loss:
                epoch_mean_loss[key].append(value if isinstance(value, (int, float)) else value.item())
            else:
                epoch_mean_loss[key] = [value if isinstance(value, (int, float)) else value.item()]


    for key, values in epoch_mean_loss.items():
        epoch_mean_loss[key] = np.mean(np.array(values))

    return epoch_mean_loss

def load_state_dict_verbose(module, state_dict, *, strict=False, label=""):
    """
    ``module.load_state_dict`` wrapper that loudly reports missing / unexpected keys.

    Loading planner / reward / predictor checkpoints across stages with ``strict=False``
    is intentional (some heads are optional), but silent key drops have masked typos in
    ``--hidden_dim`` / ``--num_heads`` etc. in the past. This wrapper makes the same
    mismatches visible at load time.
    """
    res = module.load_state_dict(state_dict, strict=strict)
    missing = getattr(res, "missing_keys", []) or []
    unexpected = getattr(res, "unexpected_keys", []) or []
    tag = f"[{label}] " if label else ""
    if missing:
        print(f"{tag}WARNING missing_keys ({len(missing)}): {missing[:8]}"
              f"{' ...' if len(missing) > 8 else ''}")
    if unexpected:
        print(f"{tag}WARNING unexpected_keys ({len(unexpected)}): {unexpected[:8]}"
              f"{' ...' if len(unexpected) > 8 else ''}")
    return res


def save_model(model, optimizer, scheduler, save_path, epoch, train_loss, wandb_id, ema, is_best=False):
    """
    save the model to path

    Writes ``model_epoch_<n>_trainloss_<l>.pth`` and ``latest.pth`` unconditionally.
    When ``is_best`` is True, also writes ``best.pth`` (kept stable across non-improving
    periodic saves so downstream stages can chain off the best checkpoint reliably).
    """
    save_model = {'epoch': epoch + 1,
                  'model': model.state_dict(),
                  'ema_state_dict': ema.state_dict(),
                  'optimizer': optimizer.state_dict(),
                  'schedule': scheduler.state_dict(),
                  'loss': train_loss,
                  'wandb_id': wandb_id}

    with io.BytesIO() as f:
        torch.save(save_model, f)
        payload = f.getvalue()
        fileio.put(payload, f'{save_path}/model_epoch_{epoch+1}_trainloss_{train_loss:.4f}.pth')
        fileio.put(payload, f"{save_path}/latest.pth")
        if is_best:
            fileio.put(payload, f"{save_path}/best.pth")

def resume_model(path: str, model, optimizer, scheduler, ema, device):
    """
    load ckpt from path
    """
    path = os.path.join(path, 'latest.pth')
    ckpt = fileio.get(path)
    with io.BytesIO(ckpt) as f:
        ckpt = torch.load(f, weights_only=False)

    # load model
    try:
        model.load_state_dict(ckpt['model'])
    except:
        model.load_state_dict(ckpt)                   
    print("Model load done")
    
    # load optimizer
    try:
        optimizer.load_state_dict(ckpt['optimizer'])
        print("Optimizer load done")
    except:
        print("no pretrained optimizer found")
            
    # load schedule
    try:
        scheduler.load_state_dict(ckpt['schedule'])
        print("Schedule load done")
    except:
        print("no schedule found,")
    
    # load step
    try:
        init_epoch = ckpt['epoch']
        print("Step load done")
    except:
        init_epoch = 0

    # Load wandb id
    try:
        wandb_id = ckpt['wandb_id']
        print("wandb id load done")
    except:
        wandb_id = None

    try:
        ema.ema.load_state_dict(ckpt['ema_state_dict'])
        ema.ema.eval()
        for p in ema.ema.parameters():
            p.requires_grad_(False)

        print("ema load done")
    except:
        print('no ema shadow found')

    return model, optimizer, scheduler, init_epoch, wandb_id, ema


