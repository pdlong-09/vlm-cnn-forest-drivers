# Code and predictions: VLM versus CNN for deforestation drivers from Indonesia to other continents

## Description

Companion release of the article "A fine-tuned vision-language model transfers better than convolutional networks for
attributing deforestation drivers from Indonesia to other continents".

The article compares a convolutional neural network (CNN) and a vision-language model (VLM; Qwen3.5-9B fine-tuned with
QLoRA). Both were trained on the same ForestNet events from Indonesia and read the same yearly Landsat composites, up to
four years after the loss. They were evaluated, for conversion to agriculture, on the ForestNet test set, on independently
labelled cells elsewhere in Indonesia, and on cells from Africa, Latin America and the Asia-Pacific region.

The release holds the evaluation cells with their event polygons, the predicted probabilities of every model and all the
code. The main results can be recomputed from these files alone (see Usage Instructions). Satellite images, evidence
cards and trained model weights are not included.

## Dataset Information

### Files

| File | Rows | Content |
| --- | --- | --- |
| `data/forestnet_events.csv` | 2757 | ForestNet events (Irvin et al. 2020), training data of all models |
| `data/indonesia_cells.csv` | 548 | Cells of Sims et al. (2025) in Indonesia, outside ForestNet |
| `data/e2_cells.csv` | 1500 | E2, regional validation: cells from three regions outside Indonesia |
| `data/e3_cells.csv` | 2306 | E3, held-out test: the remaining valid cells of the same regions |
| `data/{indonesia,e2,e3}_event_polygons.geojson` | one per cell | Event polygons in longitude and latitude |
| `data/cameroon_events.csv` | 2411 | Cam-ForestNet events (Debus et al. 2024), in-domain check |
| `data/codebook.csv` | 121 | Meaning of every value of the categorical columns |
| `predictions/<set>_<model>.csv` | 67 files | Class probabilities per event and step k |

### Columns

- **ForestNet** (`data/forestnet_events.csv`): `eid`; the official split (`split`); the split used in the article, with the
  validation set divided into selection and calibration halves (`part`); the detailed and four-class labels (`label`,
  `merged_label`); `latitude`, `longitude`, the year of loss (`year`); and the path of the event in the ForestNet release
  (`example_path`).
- **External cells** (`data/{indonesia,e2,e3}_cells.csv`):
  - identifiers: `eid`; `sid`, the cell number within the set, which links to the GeoJSON files; `ID` and `src`, which
    together identify the cell in the files of Sims et al.;
  - from Sims et al.: `Latitude`, `Longitude`, `Driver_primary`, `Confidence_primary`, `Tags`;
  - added by us: `dist_fn_train_km` (distance to the nearest ForestNet training event); the event year `year`, its loss area
    `year_ha` and the loss area of 2001-2022 `loss_ha_2001_2022`, in hectares from Hansen Global Forest Change; the binary
    label `label_agri` (permanent agriculture or shifting cultivation); and, for E2 and E3, `country` (LSIB 2017
    boundaries) and the region group `group`.
- **Event polygons** (`data/*_event_polygons.geojson`): the loss pixels of the event year in each 0.01-degree cell, with
  `sid` as the only property.
- **Cam-ForestNet** (`data/cameroon_events.csv` and `predictions/cameroon_{cnn_resnet18,vlm}.csv`): in-domain check
  (Landsat only). Events with the official split (`split_official`), our split by 3 km spatial clusters (`part_spatial`;
  models were trained on `fit`, selected on `holdout`, `val` calibrated the early-attribution rule, scored on `test`), the
  15-class and the 4-class labels. Probabilities of the CNN (all events; `fit` rows are in-sample) and of the fine-tuned
  VLM; score them on `part_spatial == "test"` only. The file numbers its events separately, so its `eid` values overlap
  those of ForestNet. Exported from our run outputs by `code/scripts/export_cam_release.py`; the Cam-ForestNet inputs and
  the training scripts of this check are not part of the release.
- **Predictions** (`predictions/<set>_<model>.csv`): `eid`, step `k` (years of evidence after the loss; only k = 4 where a
  model was run at k = 4 only) and the probabilities `p_plantation`, `p_smallholder`, `p_grassland`, `p_other`. Binary
  conversion to agriculture: P(plantation) + P(smallholder) >= 0.5.

`eid` links the cells to the predictions: ForestNet events are numbered from 0, and the cells of Indonesia, E2 and E3 are
100000, 200000 and 300000 plus `sid`.

`data/codebook.csv` maps the coded categories to their meaning. The binary label `label_agri` is 1 for conversion to
agriculture and 0 otherwise. The one `Tags` value that reads "Tags" (ID 7145) is in the source file, and `Tags` was not used.

### Prediction sets and models

Sets: `forestnet` (validation and test events for the VLM variants), `indonesia`, `e2`, `e3`. Models:

- `cnn_resnet18` (mean of three seeds; the same model on every set), `cnn_resnet50`, `cnn_convnext_tiny`;
  `cnn_resnet18_earlier_run`: ForestNet predictions of an earlier ResNet-18 run with the same recipe whose weights were
  not kept (reference only);
- `vlm` (fine-tuned, seed 0), `vlm_seed1`, `vlm_seed2`, `vlm_zero_shot`, `vlm_eight_examples`,
  `vlm_vision_encoder_tuned`, `vlm_image_only` and `vlm_text_only` (input ablation of the fine-tuned VLM),
  `vlm_country_named` (E3 cards whose first line names the cell's country instead of Indonesia);
- controls for the inputs: `cnn_resnet18_3km` (full 3 km chips), `cnn_resnet18_3km_factsheet` (full chips and the
  numbers of the fact sheet), `vlm_images_only` with `_seed1` and `_seed2` (VLM fine-tuned and run on cards without the
  numbers; seed 0 has no suffix);
- `lightgbm` (ForestNet model with all features), `lightgbm_crosscontinent` (without clear-observation counts);
- fusion variants on E2: `vlm_cnn_feature_tokens`, `vlm_cnn_probabilities_text`,
  `vlm_cnn_probabilities_text_neighbours`, `vlm_alphaearth_tokens_dropout*_epoch*` (four configurations; dropout 0.5,
  epoch 2 was chosen and is `e3_vlm_alphaearth_tokens`), `alphaearth_probe`.

The CNN + VLM model of the article is the mean of `cnn_resnet18` and `vlm`.

### Source data

| Dataset | Used for | Access |
| --- | --- | --- |
| ForestNet (Irvin et al. 2020) | Training events, labels, loss polygons | <https://stanfordmlgroup.github.io/projects/forestnet/> |
| Reference data of Sims et al. (2025) | External cells and labels | <https://doi.org/10.5281/zenodo.15224684> |
| Cam-ForestNet (Debus et al. 2024) | In-domain check in Cameroon | See the article of Debus et al. |
| Hansen Global Forest Change v1.13 | Event years, loss areas, event polygons | Google Earth Engine, `UMD/hansen/global_forest_change_2025_v1_13` |
| Landsat 5/7/8/9 Collection 2 Level 2 | Yearly composites | Google Earth Engine, `LANDSAT/{LT05,LE07,LC08,LC09}/C02/T1_L2` |
| SRTM | Elevation | Google Earth Engine, `USGS/SRTMGL1_003` |
| AlphaEarth Foundations embeddings | Fusion variants | Google Earth Engine, `GOOGLE/SATELLITE_EMBEDDING/V1/ANNUAL` |
| LSIB 2017 boundaries | Country of each cell | Google Earth Engine, `USDOS/LSIB_SIMPLE/2017` |

## Code Information

The code is written in Python, with one Bash script.

| Path | Content |
| --- | --- |
| `code/scripts/reproduce_main_results.py` | Recomputes the main results from this release |
| `code/rsdrivers/` | Package used by the scripts: reading the data, Earth Engine requests, evidence cards, tabular features, learn-then-test |
| `code/scripts/` | Run on the local machine: data construction, GPU batch drivers, tabular models, evaluation, figures and tables |
| `code/server/` | Run on the GPU machines: CNN and QLoRA training, inference, resumable pipelines, CPU robustness analyses |
| `code/gpu_queue.sh` | Queue that runs jobs one at a time on a shared GPU |

The drivers ran the GPU stages on two remote machines; their host names and paths are replaced by placeholders (`server1`,
`server2`, `/home/user`), and passwords are read from the environment and are not included.

Comments that mention "design notes" refer to the authors' internal analysis plan, written before the scoring runs; items
marked "post hoc" were added afterwards. The plan was not publicly registered and is not part of the release. Labels such
as P4-1 or P6-2 in the evaluation scripts name the comparisons of that plan, and section signs (for example "§4-5") point
to its sections. `rsdrivers/earlystop_v2.py` also implements options that the article does not use (a year-decreasing
threshold and temperature scaling); the article's results use beta = 0 and no temperature scaling. `rsdrivers/landsat.py`
is an earlier route through Microsoft Planetary Computer that the scripts do not use.

## Usage Instructions

### 1. Recompute the main results

From the root of this repository:

```
pip install numpy pandas
python code/scripts/reproduce_main_results.py
```

This needs no images, GPU or Earth Engine account, and it prints:

```
ForestNet test, four classes (n = 668): macro-F1 CNN 0.728, VLM 0.716, CNN + VLM 0.762

Balanced accuracy of conversion to agriculture at k = 4
ForestNet test (n = 668): CNN 0.840, VLM 0.834, difference -0.006 (95% interval, cells -0.0423 to 0.0309)
Indonesia outside ForestNet (n = 548): CNN 0.730, VLM 0.725, difference -0.005 (95% interval, cells -0.0488 to 0.0361)
E2, regional validation (n = 1500): CNN 0.651, VLM 0.707, difference +0.057 (95% interval, cells 0.0281 to 0.0845; countries 0.0294 to 0.0858)
E3, held-out test (n = 2306): CNN 0.649, VLM 0.703, difference +0.054 (95% interval, cells 0.0318 to 0.0767; countries 0.0289 to 0.0829)
```

### 2. Use the predictions

```python
import pandas as pd

cells = pd.read_csv("data/e3_cells.csv")
vlm = pd.read_csv("predictions/e3_vlm.csv")
vlm = vlm[vlm.k == 4].merge(cells[["eid", "label_agri", "country"]], on="eid")
vlm["pred_agri"] = (vlm.p_plantation + vlm.p_smallholder >= 0.5).astype(int)
```

### 3. Check the files

```
sha256sum -c SHA256SUMS.txt
```

### 4. Re-run the pipeline

A full re-run needs the source data, an Earth Engine account, GPUs and adaptation to your machines. The commands in the
script headers were written for the authors' project folder, so recreate this layout under a folder `<root>`:

```
<root>/paper/rs_forestnet/{rsdrivers,scripts,server}/        contents of code/
<root>/tools/gpu_queue.sh                                    code/gpu_queue.sh
<root>/data/raw/forestnet_A/deep/downloads/ForestNetDataset/ ForestNet release
<root>/data/raw/sims2025/                                    validation_2001_2022.csv, training_2001_2022_v1.csv
<root>/.env                                                  GEE_PROJECT=<your Earth Engine project>
```

Scripts write to `<root>/data/interim/` and `<root>/runs/`; every script header gives its command, inputs and outputs.
Run them from `<root>` in this order:

| Step | Scripts | Result |
| --- | --- | --- |
| 1. ForestNet tables | The Python lines below | `parts.csv`, `static_features.parquet` |
| 2. ForestNet evidence | `build_series_gee.py`, `build_chips_gee.py` | Index series and yearly chips |
| 3. External cells | `sims_build.py --pool idn\|trop\|e3 --what events\|series\|static\|chips` | Cells, polygons, series, static values, chips |
| 4. AlphaEarth embeddings | `aef_v5.py fetch`, `aef_v5.py pack`, `probe_v5.py` | Embeddings and probe |
| 5. Tabular model | `m1_lightgbm.py` | LightGBM probabilities |
| 6. CNN | `m2_cnn.py`, `fn_v3_gpu.py` | CNN weights and probabilities |
| 7. VLM | `m3_vlm.py`, `m3c_qlora.py` | Adapters and VLM probabilities |
| 8. Fusion and E3 | `fn_v4.py`, `fn_v5.py` | Fusion variants, E2 choice, E3 scoring |
| 9. Robustness | `fn_robust_a.py`, `fn_fair_a.py`, `fn_country_ablation.py` | Other backbones, ablation, input controls |
| 10. Evaluation | `evaluate*.py`, `fair_seeds_eval.py`, `early_stop_source.py`, `early_stop_split.py`, `label_sensitivity.py`, `crop_coverage.py` | JSON result files |
| 11. Figures and tables | `make_figures_a.py`, `make_supp_a.py` | Figures and supplementary tables |

No script writes the two tables of step 1. These lines rebuild them exactly:

```python
import sys

sys.path.insert(0, "paper/rs_forestnet")
import pandas as pd
from rsdrivers.features import static_features, val_split
from rsdrivers.io import INTERIM, events

ev = events()
INTERIM.mkdir(parents=True, exist_ok=True)
pd.DataFrame({"eid": ev.eid, "part": val_split(ev)}).to_csv(INTERIM / "parts.csv", index=False)
pd.DataFrame([{"eid": r.eid, **static_features(r)} for r in ev.itertuples()]).to_parquet(INTERIM / "static_features.parquet")
```

Two limits apply:

- `sims_build.py --pool idn` reads `data/raw/sims2025/sims_idn_plots_hansen.csv`, a list of the Sims cells in Indonesia that
  was prepared outside these scripts and is not released. The valid cells it gave are in `data/indonesia_cells.csv`.
- The drivers in `code/scripts/` that start GPU batches use the ssh of Git for Windows and the host names `lab` and
  `server2`, through `rsdrivers/lab.py`. Adapt that file to your hosts, or run the scripts of `code/server/` directly on a
  GPU machine, for example:

```
python cnn_train.py --data chips.npz --out probs.npz --epochs 40 --seed 0
python qlora_train.py --model <Qwen3.5-9B dir> --train train.jsonl --images <dir> --out <adapter dir> --epochs 2
python vlm_infer.py --model <Qwen3.5-9B dir> --input batch.jsonl --images <dir> --output out.jsonl --lora <adapter dir>
```

The base model is at <https://huggingface.co/Qwen/Qwen3.5-9B>. The expected inputs are described in each script header.

## Requirements

| Task | Software | Hardware |
| --- | --- | --- |
| Recompute the main results | Python 3.12, numpy 2.5.3, pandas 3.0.6 | Any computer |
| Data construction, tabular models, evaluation | Python 3.12, numpy 2.5.4, pandas 3.0.6, scipy 1.18.1, scikit-learn 1.9.1, lightgbm 4.7.0, shapely 2.2.0, pillow 13.0.0, matplotlib 3.11.2, pyarrow 26.0.0, earthengine-api 1.7.49; a Google Earth Engine project | CPU; Windows 11 in our runs |
| CNN training | Python 3.12, PyTorch 2.13.0, torchvision 0.28.0 (CUDA 13.0 builds), numpy, scikit-learn | NVIDIA GPU with 12 GB; Linux |
| VLM training and inference | Python 3.12, PyTorch, a Transformers release with Qwen3.5 support, PEFT 0.21.2, bitsandbytes, vLLM 0.31.0 | NVIDIA RTX 4090 (24 GB); Linux |

The versions listed are those of our runs. Fine-tuning for two epochs took about 46 minutes, and inference on E3 about
13 minutes per seed, on one RTX 4090.

## Methodology

1. **Events.** ForestNet events keep their official split, and the validation set is divided into a selection half and a
   calibration half. In each 0.01-degree cell of the reference data of Sims et al., the event is the year of 2001-2022 with
   the largest Hansen loss (at least 0.5 ha), kept if it falls in 2001-2020 so that four later years exist. Its polygon is
   the loss pixels of that year.
2. **Labels.** Plantation and smallholder agriculture (ForestNet), and permanent agriculture and shifting cultivation (Sims
   et al.), count as conversion to agriculture.
3. **Preprocessing.** Landsat scenes are cloud-masked and reduced to one median composite per calendar year, from two years
   before the loss to four years after it. Each year gives a false-colour chip of 3 km x 3 km and index medians inside the
   loss polygon and in a 150 m ring.
4. **Models.** At step k, a model sees only the evidence up to year k. The CNN reads stacked chips. The VLM reads an evidence
   card made of an image grid and a fact sheet, and it answers a four-option question; its class probabilities are the
   letter probabilities averaged over four cyclic option orders. Neither model is tuned on data from other continents.
5. **Evaluation.** Macro-F1 on the four ForestNet classes and balanced accuracy of conversion to agriculture at k = 4, with
   paired bootstrap intervals over cells and over countries. E2 served for comparisons and the choice of the fusion
   model. All models and settings were fixed before E3 was scored.
6. **Early attribution.** A learn-then-test rule stops at the first year whose highest class probability reaches a
   calibrated threshold, with the threshold calibrated on ForestNet or on local labels.

The article and its Supplemental File S1 describe each step in full.

## Citations

Please cite the article, this archive and the data sources.

- Nguyen Dinh Manh Linh, Pham Duy Long, Nguyen Duc Anh, Do Nhat Quang. A fine-tuned vision-language model transfers better
  than convolutional networks for attributing deforestation drivers from Indonesia to other continents. Manuscript
  submitted for publication.
- Code and data archive: <https://doi.org/10.5281/zenodo.23799446> (all versions).
- Brown CF et al. 2025. AlphaEarth Foundations: an embedding field model for accurate and efficient global mapping from
  sparse label data. ArXiv preprint. <https://doi.org/10.48550/arXiv.2507.22291>
- Crawford CJ et al. 2023. The 50-year Landsat collection 2 archive. Science of Remote Sensing 8:100103.
  <https://doi.org/10.1016/j.srs.2023.100103>
- Debus A et al. 2024. A labelled dataset to classify direct deforestation drivers from Earth Observation imagery in
  Cameroon. Scientific Data 11(1):564. <https://doi.org/10.1038/s41597-024-03384-z>
- Farr TG et al. 2007. The Shuttle Radar Topography Mission. Reviews of Geophysics 45(2):RG2004.
  <https://doi.org/10.1029/2005RG000183>
- Hansen MC et al. 2013. High-resolution global maps of 21st-century forest cover change. Science 342(6160):850-853.
  <https://doi.org/10.1126/science.1244693>
- Irvin J et al. 2020. ForestNet: classifying drivers of deforestation in Indonesia using deep learning on satellite
  imagery. NeurIPS 2020 Workshop on Tackling Climate Change with Machine Learning.
  <https://doi.org/10.48550/arXiv.2011.05479>
- Sims MJ et al. 2025. Global drivers of forest loss at 1 km resolution. Environmental Research Letters 20(7):074027.
  <https://doi.org/10.1088/1748-9326/add606>

## License & Contribution Guidelines

Code: MIT licence. Data and predictions: CC BY 4.0, as the inputs from which they derive; please cite these sources
together with the article:
- ForestNet (Irvin et al. 2020), CC BY 4.0: event labels, coordinates and split;
- reference data of Sims et al. (2025), CC BY 4.0: cell labels, confidence and coordinates;
- Cam-ForestNet (Debus et al. 2024), CC BY 4.0: labels, coordinates and years only (no NICFI imagery, no OpenStreetMap
  content);
- Hansen et al. (2013) Global Forest Change v1.13, CC BY 4.0: event years, loss areas and event polygons were derived
  from its loss layer;
- the model inputs also used Landsat Collection 2 (courtesy of the U.S. Geological Survey), SRTM elevation and the
  AlphaEarth Foundations Satellite Embedding dataset (produced by Google and Google DeepMind), accessed through Google
  Earth Engine; none of these images or embeddings is redistributed here.

Changes made: the source labels were mapped to the four ForestNet classes and to the binary label `label_agri`, events
were derived per 0.01-degree cell, and our own splits and model predictions were added.

This repository is the record of the article, so its results are not updated with new contributions. Questions and bug
reports are welcome as GitHub issues.

## Development

The code was written with the help of an AI coding assistant (Claude, Anthropic), as stated in the article; the
authors designed the analyses and checked every result against the data.
