# Experiment Report: phaseaware_best_models

- dataset: `data/wv4-formal-hitran-standard-6000`
- device: `cuda`

| kind | run | model | window | split | loss | MAE | RMSE | R2 | x_N2 R2 | Aitchison mean | sum abs error |
|---|---|---|---|---|---:|---:|---:|---:|---:|---:|---:|
| ml | ridge_all_modalities | ridge | full | train |  | 2.582798 | 3.875644 | 0.805045 | 0.243345 |  |  |
| ml | ridge_all_modalities | ridge | full | val |  | 2.621623 | 3.930793 | 0.797877 | 0.231498 |  |  |
| ml | ridge_all_modalities | ridge | full | test |  | 2.650242 | 3.981043 | 0.796795 | 0.217338 |  |  |
| ml | ridge_all_modalities | ridge | full | extrapolation |  | 2.628639 | 3.960391 | 0.796587 | 0.227272 |  |  |
| ml | ridge_all_modalities_phase_exposure | ridge | phase:exposure | train |  | 2.360117 | 3.292109 | 0.859332 | 0.640705 |  |  |
| ml | ridge_all_modalities_phase_exposure | ridge | phase:exposure | val |  | 2.361556 | 3.318495 | 0.855942 | 0.650228 |  |  |
| ml | ridge_all_modalities_phase_exposure | ridge | phase:exposure | test |  | 2.440891 | 3.471157 | 0.845514 | 0.601431 |  |  |
| ml | ridge_all_modalities_phase_exposure | ridge | phase:exposure | extrapolation |  | 2.327840 | 3.276143 | 0.860803 | 0.647007 |  |  |
| ml | ridge_all_modalities_phase_recovery | ridge | phase:recovery | train |  | 2.218834 | 3.104467 | 0.874910 | 0.601102 |  |  |
| ml | ridge_all_modalities_phase_recovery | ridge | phase:recovery | val |  | 2.250696 | 3.152986 | 0.869953 | 0.601716 |  |  |
| ml | ridge_all_modalities_phase_recovery | ridge | phase:recovery | test |  | 2.302206 | 3.257759 | 0.863925 | 0.576721 |  |  |
| ml | ridge_all_modalities_phase_recovery | ridge | phase:recovery | extrapolation |  | 2.312697 | 3.278043 | 0.860642 | 0.588571 |  |  |
| ml | ridge_all_modalities_early_050 | ridge | early:0.50 | train |  | 2.738813 | 3.809651 | 0.811627 | 0.480207 |  |  |
| ml | ridge_all_modalities_early_050 | ridge | early:0.50 | val |  | 2.718687 | 3.791132 | 0.811985 | 0.481873 |  |  |
| ml | ridge_all_modalities_early_050 | ridge | early:0.50 | test |  | 2.875538 | 4.031505 | 0.791611 | 0.429284 |  |  |
| ml | ridge_all_modalities_early_050 | ridge | early:0.50 | extrapolation |  | 2.706420 | 3.821151 | 0.810638 | 0.491101 |  |  |
| ml | ridge_all_modalities_early_075 | ridge | early:0.75 | train |  | 2.344604 | 3.474649 | 0.843300 | 0.390199 |  |  |
| ml | ridge_all_modalities_early_075 | ridge | early:0.75 | val |  | 2.337034 | 3.450189 | 0.844281 | 0.401858 |  |  |
| ml | ridge_all_modalities_early_075 | ridge | early:0.75 | test |  | 2.453921 | 3.616618 | 0.832295 | 0.346452 |  |  |
| ml | ridge_all_modalities_early_075 | ridge | early:0.75 | extrapolation |  | 2.320204 | 3.463603 | 0.844418 | 0.399335 |  |  |
| dl | cnn1d_tcn_fusion | cnn1d_tcn_fusion | full | val | 22.463543 | 3.544690 | 4.740016 | 0.706090 | -0.012201 |  | 0.000002 |
| dl | cnn1d_tcn_fusion | cnn1d_tcn_fusion | full | test | 22.180845 | 3.520754 | 4.718535 | 0.714533 | -0.006442 |  | 0.000002 |
| dl | cnn1d_tcn_fusion | cnn1d_tcn_fusion | full | extrapolation | 21.813298 | 3.441293 | 4.674148 | 0.716660 | -0.003661 |  | 0.000002 |
| dl | cnn1d_tcn_fusion_phase_exposure | cnn1d_tcn_fusion | phase:exposure | val | 23.975465 | 3.746173 | 4.891109 | 0.687054 | -0.007887 |  | 0.000002 |
| dl | cnn1d_tcn_fusion_phase_exposure | cnn1d_tcn_fusion | phase:exposure | test | 24.317660 | 3.768798 | 4.935981 | 0.687617 | -0.000429 |  | 0.000002 |
| dl | cnn1d_tcn_fusion_phase_exposure | cnn1d_tcn_fusion | phase:exposure | extrapolation | 23.758611 | 3.714829 | 4.880065 | 0.691145 | 0.008997 |  | 0.000002 |
| dl | cnn1d_tcn_fusion_phase_recovery | cnn1d_tcn_fusion | phase:recovery | val | 24.532962 | 3.755953 | 4.961347 | 0.678001 | -0.000154 |  | 0.000002 |
| dl | cnn1d_tcn_fusion_phase_recovery | cnn1d_tcn_fusion | phase:recovery | test | 26.126797 | 3.849912 | 5.126637 | 0.663019 | -0.003105 |  | 0.000002 |
| dl | cnn1d_tcn_fusion_phase_recovery | cnn1d_tcn_fusion | phase:recovery | extrapolation | 23.775780 | 3.644084 | 4.890230 | 0.689857 | 0.005029 |  | 0.000002 |
| dl | cnn1d_tcn_fusion_early_050 | cnn1d_tcn_fusion | early:0.50 | val | 25.830506 | 3.895878 | 5.067998 | 0.664009 | 0.002260 |  | 0.000002 |
| dl | cnn1d_tcn_fusion_early_050 | cnn1d_tcn_fusion | early:0.50 | test | 26.267423 | 3.937914 | 5.135239 | 0.661887 | -0.014244 |  | 0.000002 |
| dl | cnn1d_tcn_fusion_early_050 | cnn1d_tcn_fusion | early:0.50 | extrapolation | 25.867112 | 3.893992 | 5.088551 | 0.664191 | -0.000243 |  | 0.000002 |
| dl | cnn1d_tcn_fusion_early_075 | cnn1d_tcn_fusion | early:0.75 | val | 20.393215 | 3.343795 | 4.516668 | 0.733135 | -0.004518 |  | 0.000002 |
| dl | cnn1d_tcn_fusion_early_075 | cnn1d_tcn_fusion | early:0.75 | test | 21.532905 | 3.434754 | 4.649373 | 0.722841 | -0.005361 |  | 0.000002 |
| dl | cnn1d_tcn_fusion_early_075 | cnn1d_tcn_fusion | early:0.75 | extrapolation | 20.695661 | 3.359696 | 4.557784 | 0.730592 | 0.007923 |  | 0.000002 |
