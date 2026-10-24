# FedKG-Brain

An interactive platform for brain-connectivity analysis with graph neural networks and federated learning,
built during a summer internship at ENSI (University of Manouba).

**Input:** a resting-state fMRI scan, as a region time series (`.1D` file: time points × 200 regions of the CC200 atlas).
**Output:** the probability of autism, the brain networks that drove the decision, and a 3-D view of the most
influential connections.

The platform also runs **federated training** live (each acquisition site trains locally and only model weights
are shared) and predicts how a subject's **connectome evolves** within the scan.

> Research prototype. Not a medical device and not a diagnostic tool.

## What happens to a scan

| Step | What the platform does |
|---|---|
| 1. Read the scan | loads the time series of the 200 regions |
| 2. Functional connectivity | Pearson correlation between every pair of regions (19,900 connections) |
| 3. Brain graph | keeps the 20 strongest connections of each region |
| 4. Graph neural network | 2-layer GCN (64 units, mean pooling) → probability of autism |
| 5. Explanation | gradient × input on every connection, summed per pair of functional networks |

## Tabs

- **Analyze a scan**: upload a `.1D` file (or pick a held-out subject) and follow each step live.
- **Federated training**: FedAvg across the ABIDE sites, compared with each site alone and with pooled data.
- **Connectivity evolution**: from the first two thirds of a scan, predict the connectome of the last third.
- **Model**: the pipeline and the logged results.

## Run it

```bash
pip install -r requirements.txt
python app.py --root /path/to/fedkg_brain           # opens on http://127.0.0.1:7860
python app.py --root /path/to/fedkg_brain --share   # public link (e.g. from Google Colab)
```

On first start the platform builds a connectivity cache and trains the model once (a few minutes on a GPU);
later starts take seconds. `--fast` trains for fewer epochs (quick test only).

### Data (not included in this repository)

ABIDE I preprocessed with CPAC (band-pass filtering, no global signal regression), CC200 atlas, from the
Preprocessed Connectomes Project. Expected layout under `--root`:

```
data/raw/ABIDE_PCP/ABIDE_pcp/cpac/filt_noglobal/*_rois_cc200.1D
data/raw/ABIDE_PCP/ABIDE_pcp/Phenotypic_V1_0b_preprocessed1.csv
data/raw/atlases/cc200_roi_atlas.nii.gz        (optional: 3-D brain view)
data/processed/roi_to_network.json             (optional: functional-network labels)
```

`results_real/` is created automatically for the caches and the trained model.

## Record a demo video

With the platform running:

```bash
pip install playwright && playwright install chromium
python record_demo.py --url http://127.0.0.1:7860 --upload /path/to/a_held_out_subject_rois_cc200.1D --out demo.mp4
```

The captions in the video come from the platform itself (turn them off with the *Live captions* switch).

## Results (ABIDE I, 871 subjects, 20 sites)

| Model | Accuracy | AUC | Protocol |
|---|---|---|---|
| Logistic regression, raw edges | 0.680 ± 0.032 | 0.736 ± 0.026 | 5-fold CV |
| GCN, raw graphs | 0.650 ± 0.019 | 0.694 ± 0.020 | 5-fold CV |
| GCN + knowledge graph (6 relations) | 0.643 ± 0.023 | 0.701 ± 0.029 | 5-fold CV |
| FedAvg KG-GCN, 18 sites | 0.547 | 0.595 | hold-out, 150 subjects |

Knowledge-graph priors used as node features did not improve the GCN: with one-hot region features the model can
already learn one vector per region, which absorbs any fixed per-region prior.

## Data credits

ABIDE I: Di Martino et al., *Molecular Psychiatry* 19 (2014) 659–667.
Preprocessed Connectomes Project: Craddock et al., *Frontiers in Neuroinformatics* (2013).
