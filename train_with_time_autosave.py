# =====================================================================
# File: examples/jupyter_time_autosave.py  (how to integrate in your loop)
# =====================================================================
from __future__ import annotations
from typing import Optional, Iterable
import torch
from torch.utils.data import DataLoader
from system.utils.timecheckpointer import Checkpointer, SaveConfig, TrainProgress

def train_with_time_autosave(
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler: Optional[object],
    train_loader: DataLoader,
    device: torch.device,
    *,
    ckpt_dir: str = "./ckpt_local",
    run_name: str = "run1",
    save_every_secs: int = 300,    # 5 minutes
    use_amp: bool = True,
    max_epochs: int = 999999999,   # epochs don't matter; stop when you want
) -> None:
    scaler = torch.cuda.amp.GradScaler(enabled=use_amp)
    ckpt = Checkpointer(SaveConfig(ckpt_dir=ckpt_dir, run_name=run_name,
                                   save_every_secs=save_every_secs, keep_last=5,
                                   with_amp=use_amp, verbose=True))

    # Resume if exists
    progress = ckpt.load(model=model, optimizer=optimizer, scheduler=scheduler, scaler=scaler, map_location="cpu")
    start_epoch = progress.epoch
    skip_batches = max(progress.batch_idx, -1)

    model.to(device)
    try:
        for epoch in range(start_epoch, max_epochs):
            progress.epoch = epoch
            model.train(True)

            # Fast-forward within the epoch to skip already processed batches
            data_iter = iter(train_loader)
            for _ in range(skip_batches + 1):
                if skip_batches < 0: break
                try: next(data_iter)
                except StopIteration: data_iter = iter([]); break
            skip_batches = -1

            for batch_idx, batch in enumerate(data_iter):
                inputs, targets = batch[0].to(device), batch[1].to(device)
                with torch.cuda.amp.autocast(enabled=use_amp):
                    outputs = model(inputs)
                    loss = torch.nn.functional.cross_entropy(outputs, targets)

                optimizer.zero_grad(set_to_none=True)
                if use_amp:
                    scaler.scale(loss).step(optimizer); scaler.update()
                else:
                    loss.backward(); optimizer.step()
                if scheduler is not None: scheduler.step()

                progress.batch_idx = batch_idx
                progress.global_step += 1

                # ---- TIME-BASED AUTOSAVE (every ~5 minutes) ----
                if ckpt.should_save():
                    ckpt.save(model=model, optimizer=optimizer, scheduler=scheduler,
                              scaler=scaler, progress=progress,
                              extra={"loss": float(loss.detach().cpu())})

            # End-of-epoch durable save (optional but recommended)
            progress.batch_idx = -1
            ckpt.save(model=model, optimizer=optimizer, scheduler=scheduler,
                      scaler=scaler, progress=progress)

    except KeyboardInterrupt:
        # Save on manual stop so you can resume later
        ckpt.save(model=model, optimizer=optimizer, scheduler=scheduler,
                  scaler=scaler, progress=progress)
        print("[train] Interrupted -> state saved. Re-run the cell to continue.")

# =====================================================================
# Minimal usage in your notebook
# =====================================================================
# from examples.jupyter_time_autosave import train_with_time_autosave
# train_with_time_autosave(
#     model, optimizer, scheduler, train_loader, device,
#     ckpt_dir="./checkpoints", run_name="exp1", save_every_secs=300, use_amp=True,
# )
