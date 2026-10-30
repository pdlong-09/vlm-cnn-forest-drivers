# Code and predictions: VLM versus CNN for deforestation drivers from Indonesia to other continents

Companion release of the article "A fine-tuned vision-language model transfers better than convolutional networks for
attributing deforestation drivers from Indonesia to other continents".

## Contents

- `data/forestnet_events.csv`: the 2757 ForestNet events (Irvin et al. 2020, CC BY 4.0) with the official split, the
  selection/calibration halves of the validation set (`part`) and the paths of the ForestNet release.
- `data/{indonesia,e2,e3}_cells.csv`: the evaluation cells drawn from the reference data of Sims et al. (2025, CC BY 4.0),
  with the event year and loss area derived from Hansen Global Forest Change, the binary label (`label_agri`: permanent
  agriculture or shifting cultivation) and, for E2/E3, the country and region group. `eid` links to the predictions.
- `data/*_event_polygons.geojson`: event polygons (loss pixels of the event year in each 0.01-degree cell).
- `data/cameroon_events.csv` and `predictions/cameroon_{cnn_resnet18,vlm}.csv`: in-domain check on Cam-ForestNet
  (Debus et al. 2024, CC BY 4.0; Landsat only). Events with the official split (`split_official`), our split by 3 km
  spatial clusters (`part_spatial`; models were trained on `fit`, selected on `holdout`, `val` calibrated the early-attribution rule, scored on `test`), the 15-class
  and the 4-class labels. Probabilities of the CNN (all events; `fit` rows are in-sample) and of the fine-tuned VLM; score
  them on `part_spatial == "test"` only. Exported from our run outputs by `code/scripts/export_cam_release.py`; the
  Cam-ForestNet inputs and the training scripts of this check are not part of the release.
- `predictions/<set>_<model>.csv`: class probabilities (plantation, smallholder, grassland, other) per event and step k
  (years of evidence after the loss; only k = 4 where a model was run at k = 4 only). Sets: `forestnet` (validation and
  test events for the VLM variants), `indonesia`, `e2`, `e3`. Models:
  - `cnn_resnet18` (mean of three seeds; the same model on every set), `cnn_resnet50`, `cnn_convnext_tiny`;
    `cnn_resnet18_earlier_run`: ForestNet predictions of an earlier ResNet-18 run with the same recipe whose weights were
    not kept (reference only);
  - `vlm` (fine-tuned, seed 0), `vlm_seed1`, `vlm_seed2`, `vlm_zero_shot`, `vlm_eight_examples`,
    `vlm_vision_encoder_tuned`, `vlm_image_only` and `vlm_text_only` (input ablation of the fine-tuned VLM),
    `vlm_country_named` (E3 cards whose first line names the cell's country instead of Indonesia);
  - controls for the inputs: `cnn_resnet18_3km` (full 3 km chips), `cnn_resnet18_3km_factsheet` (full chips and the
    numbers of the fact sheet), `vlm_images_only` with `_seed1` and `_seed2` (VLM fine-tuned and run on cards without the numbers; seed 0 has no suffix);
  - `lightgbm` (ForestNet model with all features), `lightgbm_crosscontinent` (without clear-observation counts);
  - fusion variants on E2: `vlm_cnn_feature_tokens`, `vlm_cnn_probabilities_text`,
    `vlm_cnn_probabilities_text_neighbours`, `vlm_alphaearth_tokens_dropout*_epoch*` (four configurations; dropout 0.5,
    epoch 2 was chosen and is `e3_vlm_alphaearth_tokens`), `alphaearth_probe`.
  The CNN + VLM model of the article is the mean of `cnn_resnet18` and `vlm`.
- `code/`: data construction (Google Earth Engine), evidence cards, CNN and QLoRA training, inference, the learn-then-test
  early stop and all evaluation scripts. The drivers ran the GPU stages on two remote machines; their host names and
  paths are replaced by placeholders (`server1`, `server2`, `/home/user`), and passwords are read from the environment
  and are not included. Comments that mention "design notes" refer to the authors' internal analysis plan, written
  before the scoring runs; items marked "post hoc" were added afterwards. The plan was not publicly registered and is not
  part of the release. Labels such as P4-1 or P6-2 in the evaluation scripts name the comparisons of that plan, and
  section signs (for example "§4-5") point to its sections. `rsdrivers/earlystop_v2.py` also implements options that the
  article does not use (a year-decreasing threshold and temperature scaling); the article's results use beta = 0 and no
  temperature scaling.

## Development

The code was written with the help of an AI coding assistant (Claude, Anthropic), as stated in the article; the
authors designed the analyses and checked every result against the data.

## Licence

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

Binary conversion to agriculture: P(plantation) + P(smallholder) >= 0.5.
