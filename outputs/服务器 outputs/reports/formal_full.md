# Experiment Report: formal_full

- dataset: `data/wv4-formal-hitran-standard-6000`
- device: `cuda`

| kind | run | model | split | loss | MAE | RMSE | R2 | x_N2 R2 | Aitchison mean | sum abs error |
|---|---|---|---|---:|---:|---:|---:|---:|---:|---:|
| ml | ridge_slow | ridge | train |  | 3.799136 | 4.941005 | 0.683132 | 0.025828 |  |  |
| ml | ridge_slow | ridge | val |  | 3.861469 | 5.054736 | 0.665765 | 0.025614 |  |  |
| ml | ridge_slow | ridge | test |  | 3.938441 | 5.149995 | 0.659941 | 0.021444 |  |  |
| ml | ridge_slow | ridge | extrapolation |  | 3.999040 | 5.264159 | 0.640614 | -0.008964 |  |  |
| ml | ridge_all_modalities | ridge | train |  | 2.582798 | 3.875644 | 0.805045 | 0.243345 |  |  |
| ml | ridge_all_modalities | ridge | val |  | 2.621623 | 3.930793 | 0.797877 | 0.231498 |  |  |
| ml | ridge_all_modalities | ridge | test |  | 2.650242 | 3.981043 | 0.796795 | 0.217338 |  |  |
| ml | ridge_all_modalities | ridge | extrapolation |  | 2.628639 | 3.960391 | 0.796587 | 0.227272 |  |  |
| ml | ridge_alr_ch4_all_modalities | ridge | train |  | 3.408497 | 4.750484 | 0.707097 | 0.115390 | 0.730170 |  |
| ml | ridge_alr_ch4_all_modalities | ridge | val |  | 3.602671 | 4.983172 | 0.675162 | 0.063423 | 0.745724 |  |
| ml | ridge_alr_ch4_all_modalities | ridge | test |  | 3.380424 | 4.736911 | 0.712306 | 0.105762 | 0.704854 |  |
| ml | ridge_alr_ch4_all_modalities | ridge | extrapolation |  | 3.569037 | 4.978596 | 0.678547 | 0.122156 | 0.740877 |  |
| ml | ridge_ilr_n2_first_all_modalities | ridge | train |  | 3.408497 | 4.750485 | 0.707097 | 0.115390 | 0.730170 |  |
| ml | ridge_ilr_n2_first_all_modalities | ridge | val |  | 3.602672 | 4.983172 | 0.675162 | 0.063423 | 0.745724 |  |
| ml | ridge_ilr_n2_first_all_modalities | ridge | test |  | 3.380424 | 4.736911 | 0.712306 | 0.105762 | 0.704854 |  |
| ml | ridge_ilr_n2_first_all_modalities | ridge | extrapolation |  | 3.569037 | 4.978596 | 0.678547 | 0.122156 | 0.740877 |  |
| ml | dynamic_stacking_svr_all_modalities | dynamic_stacking_svr | train |  | 2.543407 | 3.731534 | 0.819273 | 0.325480 |  |  |
| ml | dynamic_stacking_svr_all_modalities | dynamic_stacking_svr | val |  | 3.389070 | 4.794952 | 0.699238 | -0.068935 |  |  |
| ml | dynamic_stacking_svr_all_modalities | dynamic_stacking_svr | test |  | 3.494574 | 5.070484 | 0.670360 | -0.152079 |  |  |
| ml | dynamic_stacking_svr_all_modalities | dynamic_stacking_svr | extrapolation |  | 3.504970 | 5.041533 | 0.670368 | -0.101597 |  |  |
| dl | cnn1d | cnn1d | val | 61.926952 | 6.423765 | 7.864914 | 0.190824 | -0.024454 |  | 8.890625 |
| dl | cnn1d | cnn1d | test | 63.492713 | 6.497366 | 7.981838 | 0.183142 | -0.033058 |  | 9.023438 |
| dl | cnn1d | cnn1d | extrapolation | 63.762005 | 6.522214 | 7.989968 | 0.172070 | 0.004532 |  | 9.093750 |
| dl | tcn | tcn | val | 28.896286 | 4.151608 | 5.381320 | 0.621180 | -0.023888 |  | 2.775391 |
| dl | tcn | tcn | test | 30.969622 | 4.289006 | 5.579615 | 0.600838 | -0.060105 |  | 2.593750 |
| dl | tcn | tcn | extrapolation | 29.520232 | 4.181366 | 5.446115 | 0.615340 | -0.033305 |  | 2.593750 |
| dl | lstm | lstm | val | 73.360838 | 6.724669 | 8.576803 | 0.037711 | -0.017477 |  | 2.099609 |
| dl | lstm | lstm | test | 74.741461 | 6.786984 | 8.664304 | 0.037483 | -0.009846 |  | 2.142578 |
| dl | lstm | lstm | extrapolation | 75.162718 | 6.801312 | 8.687006 | 0.021313 | -0.012805 |  | 2.082031 |
| dl | transformer | transformer | val | 53.877635 | 5.776959 | 7.349942 | 0.293320 | -0.000839 |  | 2.548828 |
| dl | transformer | transformer | test | 55.905567 | 5.891555 | 7.497651 | 0.279239 | -0.022081 |  | 2.525391 |
| dl | transformer | transformer | extrapolation | 52.550014 | 5.709006 | 7.259306 | 0.316570 | -0.014927 |  | 2.474609 |
| dl | patchtst | patchtst | val | nan | nan | nan | nan | nan |  | nan |
| dl | patchtst | patchtst | test | nan | nan | nan | nan | nan |  | nan |
| dl | patchtst | patchtst | extrapolation | nan | nan | nan | nan | nan |  | nan |
| dl | cnn1d_tcn_fusion | cnn1d_tcn_fusion | val | 22.497951 | 3.558515 | 4.738966 | 0.706220 | -0.007906 |  | 0.000002 |
| dl | cnn1d_tcn_fusion | cnn1d_tcn_fusion | test | 22.201013 | 3.491202 | 4.724582 | 0.713801 | -0.007451 |  | 0.000002 |
| dl | cnn1d_tcn_fusion | cnn1d_tcn_fusion | extrapolation | 21.711905 | 3.490669 | 4.664549 | 0.717822 | 0.000840 |  | 0.000002 |
| dl | cnn1d_tcn_fusion_ilr | cnn1d_tcn_fusion | val | 0.301060 | 3.583567 | 5.125388 | 0.656356 | -0.111117 | 0.752922 | 0.000003 |
| dl | cnn1d_tcn_fusion_ilr | cnn1d_tcn_fusion | test | 0.273907 | 3.566010 | 5.127155 | 0.662950 | -0.091585 | 0.737528 | 0.000002 |
| dl | cnn1d_tcn_fusion_ilr | cnn1d_tcn_fusion | extrapolation | 0.312831 | 3.489429 | 4.981272 | 0.678201 | -0.032638 | 0.760273 | 0.000002 |
