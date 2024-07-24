# Reproducing the figures for the paper Kernel-Based Evaluation of Conditional Biological Sequence Models

To reproduce the sythetic simulation study (no ProteinMPNN) experiments's
figure, run

```bash
python synthetic_exp_figures_composite.py --type=calibration
python synthetic_exp_figures_composite.py --type=cgof
```

To reproduce the synthetic experiments using ProteinMPNN models, run

```bash
python mmd_cgof_synthetic_data.py --type=cgof
python mmd_cgof_synthetic_data.py --type=calibration
```

To reproduce the plots for the whole dataset experiments, run

```bash
python mmd_cgof_superfamily.py
```

Finally, to reproduce the plots for the temperature study of ProteinMPNN models
on true data, run

```bash
python mmd_cgof_whole_cath_dataset.py
```
