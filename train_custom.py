"""
Multi-Task Training Pipeline for Illumination-Aware Crater Detection

Compares three configurations:
1. Baseline: Standard YOLO-v11n
2. Illumination-Aware: YOLO + solar gating + illumination augmentations
3. Full Multi-Task: All heads + curriculum learning
"""

import os
import sys
import yaml
import argparse
import time
from pathlib import Path
from datetime import datetime

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.cuda.amp import autocast, GradScaler

# Import custom modules
from data.sombench_loader import SombenchDataset
from models.yolov11n_custom import YOLOv11nCustom
from models.losses import MultiTaskLoss
from evaluation.metrics import IlluminationStratifiedMetrics
from augmentations.illumination_aware import IlluminationAwareAugmentations

class Trainer:
    def __init__(self, config_path, device="cuda" if torch.cuda.is_available() else "cpu"):
        # Load configuration
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)

        self.device = device
        self.experiment_name = self.config['experiment']['name']
        self.start_epoch = 0

        # Create experiment directory
        self.exp_dir = Path(f"results/experiments/{self.experiment_name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}")
        self.exp_dir.mkdir(parents=True, exist_ok=True)
        (self.exp_dir / "checkpoints").mkdir(exist_ok=True)
        (self.exp_dir / "logs").mkdir(exist_ok=True)

        # Save config
        import shutil
        shutil.copy(config_path, self.exp_dir / "config.yaml")

        print(f"\n{'='*60}")
        print(f"EXPERIMENT: {self.experiment_name}")
        print(f"{'='*60}")
        print(f"Device: {device}")
        print(f"Experiment directory: {self.exp_dir}")
        print(f"{'='*60}\n")

        # Initialize components
        self._init_model()
        self._init_data()
        self._init_loss()
        self._init_optimizer()
        self._init_metrics()

    def _init_model(self):
        """Initialize model based on config"""
        model_cfg = self.config['model']

        if model_cfg['architecture'] == 'yolov11n_custom':
            self.model = YOLOv11nCustom(
                heads=model_cfg['heads'],
                solar_gating_enabled=model_cfg.get('solar_gating', {}).get('enabled', False),
                solar_embedding_dim=model_cfg.get('solar_gating', {}).get('embedding_dim', 4)
            )
        else:
            # Standard YOLO-v11n (baseline)
            from ultralytics import YOLO
            self.model = YOLO('yolov11n.pt')

        # Move to device
        self.model = self.model.to(self.device)

        # Count parameters
        num_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        print(f"Model parameters: {num_params:,}")

    def _init_data(self):
        """Initialize data loaders"""
        data_cfg = self.config['data']

        # Training dataset
        train_dataset = SombenchDataset(
            root_dir="data/sombench_wac",
            split="train",
            img_size=data_cfg['img_size'],
            augmentations=self._get_augmentations(train=True),
            use_canonical_rotation=data_cfg['augmentations'].get('canonical_rotation', False),
            use_solar_embedding=data_cfg['augmentations'].get('solar_embedding', False)
        )

        # Validation dataset
        val_dataset = SombenchDataset(
            root_dir="data/sombench_wac",
            split="val",
            img_size=data_cfg['img_size'],
            augmentations=self._get_augmentations(train=False),
            use_canonical_rotation=False,
            use_solar_embedding=False
        )

        self.train_loader = DataLoader(
            train_dataset,
            batch_size=data_cfg['batch_size'],
            shuffle=True,
            num_workers=data_cfg['num_workers'],
            pin_memory=True,
            collate_fn=train_dataset.collate_fn
        )

        self.val_loader = DataLoader(
            val_dataset,
            batch_size=data_cfg['batch_size'],
            shuffle=False,
            num_workers=data_cfg['num_workers'],
            pin_memory=True,
            collate_fn=val_dataset.collate_fn
        )

        print(f"Train samples: {len(train_dataset)}")
        print(f"Val samples: {len(val_dataset)}")

    def _get_augmentations(self, train=True):
        """Get augmentation pipeline from config"""
        if not train:
            return None

        aug_cfg = self.config['data']['augmentations']

        return IlluminationAwareAugmentations(
            horizontal_flip=aug_cfg.get('horizontal_flip', 0.5),
            rotation=aug_cfg.get('rotation', 15),
            brightness=aug_cfg.get('brightness', 0.2),
            contrast=aug_cfg.get('contrast', 0.2),
            blur=aug_cfg.get('blur', 0.3),
            illumination_aware_flip=aug_cfg.get('illumination_aware_flip', False),
            canonical_rotation=aug_cfg.get('canonical_rotation', False)
        )

    def _init_loss(self):
        """Initialize multi-task loss"""
        loss_cfg = self.config['loss']

        self.criterion = MultiTaskLoss(
            detection_loss=loss_cfg.get('detection', 'CIoU'),
            classification_loss=loss_cfg.get('classification', 'cross_entropy'),
            morphology_loss=loss_cfg.get('morphology', 'cross_entropy'),
            solar_regression_loss=loss_cfg.get('solar_regression', 'cosine_similarity'),
            illumination_consistency_loss=loss_cfg.get('illumination_consistency', None),
            weights=loss_cfg.get('weights', {})
        ).to(self.device)

    def _init_optimizer(self):
        """Initialize optimizer and scheduler"""
        train_cfg = self.config['training']

        self.optimizer = torch.optim.AdamW(
            self.model.parameters(),
            lr=train_cfg['learning_rate'],
            weight_decay=train_cfg['weight_decay']
        )

        # Cosine annealing scheduler
        self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer,
            T_max=train_cfg['epochs'] - train_cfg['warmup_epochs'],
            eta_min=1e-6
        )

        # Mixed precision scaler
        self.scaler = GradScaler()

    def _init_metrics(self):
        """Initialize evaluation metrics"""
        self.metrics = IlluminationStratifiedMetrics()

    def train_epoch(self, epoch):
        """Train for one epoch"""
        self.model.train()

        total_loss = 0.0
        num_batches = len(self.train_loader)

        pbar = enumerate(self.train_loader)

        for batch_idx, batch in pbar:
            # Move to device
            images = batch['image'].to(self.device)
            bboxes = batch['bboxes']  # List of tensors
            labels = batch['labels']  # List of tensors
            solar_meta = {
                'azimuth': batch['solar_azimuth'].to(self.device),
                'incidence': batch['solar_incidence'].to(self.device)
            }

            # Forward pass with mixed precision
            self.optimizer.zero_grad()

            with autocast():
                outputs = self.model(images, solar_meta)

                loss_dict = self.criterion(
                    outputs,
                    bboxes,
                    labels,
                    solar_meta,
                    batch.get('morphology_labels')
                )

                total_loss_batch = loss_dict['total']

            # Backward pass
            self.scaler.scale(total_loss_batch).backward()
            self.scaler.step(self.optimizer)
            self.scaler.update()

            # Update progress bar
            total_loss += total_loss_batch.item()
            avg_loss = total_loss / (batch_idx + 1)

            if batch_idx % 10 == 0:
                print(f"Epoch {epoch} | Batch {batch_idx}/{num_batches} | Loss: {avg_loss:.4f}")

        return total_loss / num_batches

    @torch.no_grad()
    def validate(self, epoch):
        """Validate on validation set"""
        self.model.eval()

        all_predictions = []
        all_targets = []

        for batch_idx, batch in enumerate(self.val_loader):
            images = batch['image'].to(self.device)
            bboxes = batch['bboxes']
            labels = batch['labels']
            solar_meta = {
                'azimuth': batch['solar_azimuth'].to(self.device),
                'incidence': batch['solar_incidence'].to(self.device)
            }

            outputs = self.model(images, solar_meta)

            # Collect predictions and targets
            for i in range(len(images)):
                pred = {
                    'bboxes': outputs['detection'][i],
                    'morphology': outputs.get('morphology', [None])[i]
                }
                target = {
                    'bboxes': bboxes[i],
                    'labels': labels[i],
                    'solar_incidence': solar_meta['incidence'][i].item()
                }

                all_predictions.append(pred)
                all_targets.append(target)

        # Calculate metrics
        metrics = self.metrics.compute_all(all_predictions, all_targets)

        return metrics

    def save_checkpoint(self, epoch, metrics, is_best=False):
        """Save model checkpoint"""
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(),
            'metrics': metrics,
            'config': self.config
        }

        # Save latest
        torch.save(checkpoint, self.exp_dir / "checkpoints" / "latest.pt")

        # Save best
        if is_best:
            torch.save(checkpoint, self.exp_dir / "checkpoints" / "best.pt")
            print(f"✅ Saved best checkpoint (mAP50: {metrics['map50']:.4f})")

    def train(self):
        """Main training loop"""
        train_cfg = self.config['training']
        best_map50 = 0.0

        print(f"\nStarting training for {train_cfg['epochs']} epochs...\n")

        for epoch in range(self.start_epoch, train_cfg['epochs']):
            print(f"\n{'='*60}")
            print(f"EPOCH {epoch + 1}/{train_cfg['epochs']}")
            print(f"{'='*60}")

            start_time = time.time()

            # Train
            train_loss = self.train_epoch(epoch)

            # Validate
            metrics = self.validate(epoch)

            # Update scheduler
            self.scheduler.step()

            # Log
            elapsed = time.time() - start_time
            print(f"\nEpoch {epoch + 1} completed in {elapsed:.1f}s")
            print(f"Train loss: {train_loss:.4f}")
            print(f"Validation mAP50: {metrics['map50']:.4f}")
            print(f"Validation mAP50-95: {metrics['map50_95']:.4f}")

            # Save checkpoint
            is_best = metrics['map50'] > best_map50
            if is_best:
                best_map50 = metrics['map50']
            self.save_checkpoint(epoch, metrics, is_best)

            # Early stopping check
            if train_cfg['early_stopping']['enabled']:
                # Implement early stopping logic here
                pass

        print(f"\n{'='*60}")
        print(f"TRAINING COMPLETE")
        print(f"Best mAP50: {best_map50:.4f}")
        print(f"{'='*60}\n")


def main():
    parser = argparse.ArgumentParser(description="Train illumination-aware crater detector")
    parser.add_argument("--config", type=str, required=True, help="Path to config YAML")
    parser.add_argument("--device", type=str, default="cuda", help="Device (cuda/cpu)")
    parser.add_argument("--resume", type=str, default=None, help="Resume from checkpoint")

    args = parser.parse_args()

    trainer = Trainer(args.config, device=args.device)

    if args.resume:
        print(f"Resuming from checkpoint: {args.resume}")
        checkpoint = torch.load(args.resume)
        trainer.model.load_state_dict(checkpoint['model_state_dict'])
        trainer.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        trainer.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        trainer.start_epoch = checkpoint['epoch'] + 1

    trainer.train()


if __name__ == "__main__":
    main()
