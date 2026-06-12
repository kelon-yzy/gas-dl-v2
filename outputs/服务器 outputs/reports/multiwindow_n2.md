# Experiment Report: multiwindow_n2

- dataset: `data/wv4-formal-hitran-standard-6000`
- device: `cpu`

| kind | run | model | window | split | loss | MAE | RMSE | R2 | x_N2 R2 | Aitchison mean | sum abs error |
|---|---|---|---|---|---:|---:|---:|---:|---:|---:|---:|
| ml | ridge_all_modalities | ridge | full | train |  | 2.582798 | 3.875644 | 0.805045 | 0.243345 |  |  |
| ml | ridge_all_modalities | ridge | full | val |  | 2.621623 | 3.930793 | 0.797877 | 0.231498 |  |  |
| ml | ridge_all_modalities | ridge | full | test |  | 2.650242 | 3.981043 | 0.796795 | 0.217338 |  |  |
| ml | ridge_all_modalities | ridge | full | extrapolation |  | 2.628639 | 3.960391 | 0.796587 | 0.227272 |  |  |
| ml | ridge_multiwindow_all_modalities | ridge | multi:full+exp+rec | train |  | 1.494329 | 2.289628 | 0.931958 | 0.741369 |  |  |
| ml | ridge_multiwindow_all_modalities | ridge | multi:full+exp+rec | val |  | 1.512198 | 2.316105 | 0.929827 | 0.739652 |  |  |
| ml | ridge_multiwindow_all_modalities | ridge | multi:full+exp+rec | test |  | 1.542862 | 2.413339 | 0.925325 | 0.712068 |  |  |
| ml | ridge_multiwindow_all_modalities | ridge | multi:full+exp+rec | extrapolation |  | 1.521647 | 2.407465 | 0.924834 | 0.724702 |  |  |
