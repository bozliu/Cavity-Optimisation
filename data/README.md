# Data Access and Reconstruction

This repository does **not** publish the original/private simulation dataset.

Use public-source reconstruction to regenerate compatible inputs:

```bash
python scripts/reconstruct_data.py --config configs/default.yaml --profile full
```

Validation:

```bash
python -m cavity_ml validate-data --config configs/default.yaml
```

The reconstruction script creates:
- `data/raw/run_3_frequencies.csv`
- `data/raw/run_3_e_h_fields.csv`
- `data/alternate/Mode1.xlsx`
- `data/reconstruction_manifest.json` (SHA256 checksums)

## Source Basis (public references)

The synthetic reconstruction uses cylindrical cavity resonator equations and deterministic transforms inspired by:

- [1] COMSOL, *Cavity Resonators* (analytic resonance expressions), https://doc.comsol.com/6.1/doc/com.comsol.help.models.rf.cavity_resonators/cavity_resonators.html
- [2] LearnEMC, *Resonant Frequency of Rectangular and Cylindrical Cavities*, https://learnemc.com/ext/calculators/cavity_resonance/cyl-res.html

## License and Use

Generated data is synthetic and intended for reproducibility/testing of this codebase.
It is **not** an official measured benchmark and should not be cited as physical ground truth.
