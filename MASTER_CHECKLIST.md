# 📋 Master Research Project Checklist

## Phase 1: Data Preparation ✅ COMPLETE

- [x] Restructure project directories
- [x] Create data download scripts
- [x] Implement custom data loader with solar metadata
- [x] Create illumination-aware augmentations
- [ ] **Download Sombench dataset** ← DO THIS NOW
- [ ] Download Robbins crater catalog
- [ ] Download LOLA DEM
- [ ] Download lunar dome catalog

---

## Phase 2: Training Pipeline ✅ COMPLETE

- [x] Create training configurations (baseline, illumination-aware, full)
- [x] Implement multi-task training script
- [x] Create custom loss functions
- [x] Implement evaluation metrics
- [x] Create experiment tracking system
- [ ] **Run baseline training** ← NEXT AFTER DATA
- [ ] Run illumination-aware training
- [ ] Run full multi-task training

---

## Phase 3: Evaluation & Analysis

- [ ] Load all 3 trained models
- [ ] Run comparative evaluation
- [ ] Generate metrics tables
- [ ] Generate F1 vs. solar angle plots
- [ ] Validate against LOLA DEM
- [ ] Perform failure mode analysis
- [ ] Analyze morphology classification accuracy

---

## Phase 4: Visualization & Paper Writing

- [ ] Generate all paper figures:
  - [ ] Training curves (loss, mAP)
  - [ ] Comparative metrics table
  - [ ] F1 vs. solar incidence angle
  - [ ] Confusion matrices
  - [ ] Sample predictions (baseline vs. ours)
  - [ ] LOLA validation scatter plot

- [ ] Write paper sections:
  - [ ] Abstract
  - [ ] Introduction
  - [ ] Related Work
  - [ ] Methods (your augmentation + architecture)
  - [ ] Experiments
  - [ ] Results
  - [ ] Discussion
  - [ ] Conclusion

---

## Phase 5: Submission

- [ ] Choose target venue:
  - [ ] LPSC 2027 (deadline: Oct-Nov 2026)
  - [ ] ISPRS Journal
  - [ ] Planetary Science Journal
  - [ ] IEEE GRSL

- [ ] Format manuscript
- [ ] Prepare supplementary materials
- [ ] Get feedback from mentors
- [ ] Submit!

---

## 🎯 Immediate Next Actions

### Right Now (15 minutes):
```bash
# 1. Download Sombench dataset
python data/download_sombench.py

# 2. Inspect what you got
python inspect_dataset.py
```

### After Download (1-2 hours):
```bash
# 3. Test the training pipeline with debug config
python -c "
import yaml
with open('configs/train_baseline.yaml', 'r') as f:
    cfg = yaml.safe_load(f)
cfg['training']['epochs'] = 1
cfg['data']['batch_size'] = 4
with open('configs/debug.yaml', 'w') as f:
    yaml.dump(cfg, f)
"

python train_custom.py --config configs/debug.yaml --device cuda
```

### Once Debug Works (2-3 hours per model):
```bash
# 4. Train baseline model
python train_custom.py --config configs/train_baseline.yaml --device cuda

# 5. Train illumination-aware model
python train_custom.py --config configs/train_illumination_aware.yaml --device cuda

# 6. Train full multi-task model
python train_custom.py --config configs/train_full.yaml --device cuda
```

### After All Training Complete (1 hour):
```bash
# 7. Compare all models
python compare_models.py \
    --experiments \
        results/experiments/baseline_* \
        results/experiments/illumination_aware_* \
        results/experiments/full_multitask_*
```

---

## 📊 Expected Timeline

| Task | Duration | Dependencies |
|------|----------|--------------|
| Data download & inspection | 1-2 hours | - |
| Debug training (1 epoch) | 30 min | Data |
| Baseline training (100 epochs) | 2-3 hours | Debug works |
| Illumination-aware training | 3-4 hours | Baseline works |
| Full multi-task training | 4-5 hours | Illumination-aware works |
| Evaluation & comparison | 1-2 hours | All models trained |
| Figure generation | 2-3 hours | Evaluation complete |
| Paper writing | 2-3 days | All figures ready |
| **Total** | **~2 weeks** | - |

---

## 🏆 Success Metrics

Your research is successful if:

✅ **Minimum** (LPSC poster):
- Illumination-aware model beats baseline by >5% F1 on low-sun test set
- Morphology classification accuracy >70%
- Code released on GitHub

✅ **Strong** (ISPRS/PSJ paper):
- Illumination-aware model beats baseline by >10% F1 on low-sun test set
- Morphology classification accuracy >80%
- LOLA validation shows >85% agreement
- Cross-domain results (Moon→Mars)

✅ **Excellent** (Nature-tier):
- All above + novel theoretical insights
- State-of-the-art on Sombench benchmark
- Comprehensive ablation study
- Open dataset release

---

## 🆘 Getting Help

If you encounter issues:

1. **Check logs**: `results/experiments/{name}/logs/`
2. **Debug mode**: Use `configs/debug.yaml` with 1 epoch, batch_size=4
3. **GPU issues**: Monitor with `nvidia-smi`
4. **Data issues**: Run `python inspect_dataset.py`

---

**You're 90% ready to start training!** 🚀

Just need to:
1. Download the dataset
2. Run debug training
3. Launch full experiments

Let's go! 💪
