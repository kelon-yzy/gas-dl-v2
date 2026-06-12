# N2 Improvement Analysis

- run_root: `outputs/runs/formal_full`
- split: `test`

| baseline | candidate | N2 R2 gain | RMSE regression | max other R2 drop | Aitchison mean | pass |
|---|---|---:|---:|---:|---:|---|
| ridge_all_modalities | ridge_alr_ch4_all_modalities | -0.111575 | 0.755868 | 0.235974 | 0.704854 | no |
| ridge_all_modalities | ridge_ilr_n2_first_all_modalities | -0.111575 | 0.755868 | 0.235974 | 0.704854 | no |
| cnn1d_tcn_fusion | cnn1d_tcn_fusion_ilr | -0.084134 | 0.402574 | 0.088387 | 0.737528 | no |

## Protocol Windows

| baseline | candidate | group | window | N2 R2 gain | RMSE regression | max other R2 drop | Aitchison mean | pass |
|---|---|---|---|---:|---:|---:|---:|---|
| ridge_all_modalities | ridge_alr_ch4_all_modalities | per_phase | baseline | -0.083210 | 1.108953 | 0.353828 | 1.464959 | no |
| ridge_all_modalities | ridge_alr_ch4_all_modalities | per_phase | exposure | -0.316743 | 1.119212 | 0.101725 | 0.648896 | no |
| ridge_all_modalities | ridge_alr_ch4_all_modalities | per_phase | recovery | -0.329831 | 1.209944 | 0.122055 | 0.650793 | no |
| ridge_all_modalities | ridge_alr_ch4_all_modalities | per_phase | steady | -0.159159 | 0.696445 | 0.201275 | 0.792965 | no |
| ridge_all_modalities | ridge_alr_ch4_all_modalities | early | 0.25 | -0.083210 | 1.108953 | 0.353828 | 1.464959 | no |
| ridge_all_modalities | ridge_alr_ch4_all_modalities | early | 0.5 | -0.169347 | 0.912404 | 0.126018 | 0.716628 | no |
| ridge_all_modalities | ridge_alr_ch4_all_modalities | early | 0.75 | -0.113200 | 0.669135 | 0.157241 | 0.670183 | no |
| ridge_all_modalities | ridge_alr_ch4_all_modalities | early | 1.0 | -0.111575 | 0.755868 | 0.235974 | 0.704854 | no |
| ridge_all_modalities | ridge_ilr_n2_first_all_modalities | per_phase | baseline | -0.083210 | 1.108954 | 0.353828 | 1.464959 | no |
| ridge_all_modalities | ridge_ilr_n2_first_all_modalities | per_phase | exposure | -0.316743 | 1.119212 | 0.101725 | 0.648896 | no |
| ridge_all_modalities | ridge_ilr_n2_first_all_modalities | per_phase | recovery | -0.329832 | 1.209946 | 0.122055 | 0.650793 | no |
| ridge_all_modalities | ridge_ilr_n2_first_all_modalities | per_phase | steady | -0.159159 | 0.696445 | 0.201275 | 0.792965 | no |
| ridge_all_modalities | ridge_ilr_n2_first_all_modalities | early | 0.25 | -0.083210 | 1.108954 | 0.353828 | 1.464959 | no |
| ridge_all_modalities | ridge_ilr_n2_first_all_modalities | early | 0.5 | -0.169347 | 0.912405 | 0.126019 | 0.716628 | no |
| ridge_all_modalities | ridge_ilr_n2_first_all_modalities | early | 0.75 | -0.113200 | 0.669135 | 0.157241 | 0.670183 | no |
| ridge_all_modalities | ridge_ilr_n2_first_all_modalities | early | 1.0 | -0.111575 | 0.755868 | 0.235974 | 0.704854 | no |

## Conditional Bins

| baseline | candidate | group | bin | count | range | N2 R2 gain | RMSE regression | max other R2 drop | pass |
|---|---|---|---|---:|---|---:|---:|---:|---|
| ridge_all_modalities | ridge_alr_ch4_all_modalities | n2_bins | 0.025525_5.00919 | 154 | 0.025525-5.00919 | 11.466262 | -0.320393 | 0.223820 | no |
| ridge_all_modalities | ridge_alr_ch4_all_modalities | n2_bins | 14.9765_19.9602 | 142 | 14.9765-19.9602 | -12.318765 | 1.399608 | 0.217441 | no |
| ridge_all_modalities | ridge_alr_ch4_all_modalities | n2_bins | 5.00919_9.99285 | 166 | 5.00919-9.99285 | 1.560446 | 0.566495 | 0.283776 | no |
| ridge_all_modalities | ridge_alr_ch4_all_modalities | n2_bins | 9.99285_14.9765 | 138 | 9.99285-14.9765 | -6.317847 | 1.686738 | 0.217720 | no |
| ridge_all_modalities | ridge_alr_ch4_all_modalities | ch4_bins | 40_53.8513 | 97 | 40-53.8513 | -1.574456 | 1.191932 | 1.368252 | no |
| ridge_all_modalities | ridge_alr_ch4_all_modalities | ch4_bins | 53.8513_67.7026 | 192 | 53.8513-67.7026 | -0.133598 | 1.492414 | 1.480695 | no |
| ridge_all_modalities | ridge_alr_ch4_all_modalities | ch4_bins | 67.7026_81.5539 | 219 | 67.7026-81.5539 | -0.030820 | 0.649938 | 0.625053 | no |
| ridge_all_modalities | ridge_alr_ch4_all_modalities | ch4_bins | 81.5539_95.4052 | 92 | 81.5539-95.4052 | 1.012710 | -1.058153 | 0.086666 | no |
| ridge_all_modalities | ridge_ilr_n2_first_all_modalities | n2_bins | 0.025525_5.00919 | 154 | 0.025525-5.00919 | 11.466266 | -0.320394 | 0.223820 | no |
| ridge_all_modalities | ridge_ilr_n2_first_all_modalities | n2_bins | 14.9765_19.9602 | 142 | 14.9765-19.9602 | -12.318769 | 1.399609 | 0.217441 | no |
| ridge_all_modalities | ridge_ilr_n2_first_all_modalities | n2_bins | 5.00919_9.99285 | 166 | 5.00919-9.99285 | 1.560446 | 0.566495 | 0.283776 | no |
| ridge_all_modalities | ridge_ilr_n2_first_all_modalities | n2_bins | 9.99285_14.9765 | 138 | 9.99285-14.9765 | -6.317847 | 1.686738 | 0.217720 | no |
| ridge_all_modalities | ridge_ilr_n2_first_all_modalities | ch4_bins | 40_53.8513 | 97 | 40-53.8513 | -1.574456 | 1.191932 | 1.368251 | no |
| ridge_all_modalities | ridge_ilr_n2_first_all_modalities | ch4_bins | 53.8513_67.7026 | 192 | 53.8513-67.7026 | -0.133598 | 1.492414 | 1.480696 | no |
| ridge_all_modalities | ridge_ilr_n2_first_all_modalities | ch4_bins | 67.7026_81.5539 | 219 | 67.7026-81.5539 | -0.030820 | 0.649938 | 0.625053 | no |
| ridge_all_modalities | ridge_ilr_n2_first_all_modalities | ch4_bins | 81.5539_95.4052 | 92 | 81.5539-95.4052 | 1.012711 | -1.058154 | 0.086666 | no |
| cnn1d_tcn_fusion | cnn1d_tcn_fusion_ilr | n2_bins | 0.025525_5.00919 | 154 | 0.025525-5.00919 | 16.601958 | -1.212572 | 0.007422 | yes |
| cnn1d_tcn_fusion | cnn1d_tcn_fusion_ilr | n2_bins | 14.9765_19.9602 | 142 | 14.9765-19.9602 | -13.934673 | 1.263481 | 0.345137 | no |
| cnn1d_tcn_fusion | cnn1d_tcn_fusion_ilr | n2_bins | 5.00919_9.99285 | 166 | 5.00919-9.99285 | 1.752069 | -0.046280 | 0.031475 | no |
| cnn1d_tcn_fusion | cnn1d_tcn_fusion_ilr | n2_bins | 9.99285_14.9765 | 138 | 9.99285-14.9765 | -7.474700 | 1.374337 | 0.219521 | no |
| cnn1d_tcn_fusion | cnn1d_tcn_fusion_ilr | ch4_bins | 40_53.8513 | 97 | 40-53.8513 | -1.954309 | 2.009477 | 5.247519 | no |
| cnn1d_tcn_fusion | cnn1d_tcn_fusion_ilr | ch4_bins | 53.8513_67.7026 | 192 | 53.8513-67.7026 | -0.174484 | 0.331308 | 0.468112 | no |
| cnn1d_tcn_fusion | cnn1d_tcn_fusion_ilr | ch4_bins | 67.7026_81.5539 | 219 | 67.7026-81.5539 | 0.154400 | -0.187645 | 0.065158 | no |
| cnn1d_tcn_fusion | cnn1d_tcn_fusion_ilr | ch4_bins | 81.5539_95.4052 | 92 | 81.5539-95.4052 | 1.205057 | -0.892024 | -0.159271 | yes |

## Protocol Conditional Bins

| baseline | candidate | window group | window | bin group | bin | count | range | N2 R2 gain | RMSE regression | max other R2 drop | pass |
|---|---|---|---|---|---|---:|---|---:|---:|---:|---|
| ridge_all_modalities | ridge_alr_ch4_all_modalities | per_phase | baseline | n2_bins | 0.025525_5.00919 | 154 | 0.025525-5.00919 | 12.328611 | -0.712058 | 0.221167 | no |
| ridge_all_modalities | ridge_alr_ch4_all_modalities | per_phase | baseline | n2_bins | 14.9765_19.9602 | 142 | 14.9765-19.9602 | -13.405497 | 2.703215 | 1.381640 | no |
| ridge_all_modalities | ridge_alr_ch4_all_modalities | per_phase | baseline | n2_bins | 5.00919_9.99285 | 166 | 5.00919-9.99285 | 2.617941 | 0.372992 | 0.154826 | no |
| ridge_all_modalities | ridge_alr_ch4_all_modalities | per_phase | baseline | n2_bins | 9.99285_14.9765 | 138 | 9.99285-14.9765 | -5.236745 | 1.952527 | 0.857286 | no |
| ridge_all_modalities | ridge_alr_ch4_all_modalities | per_phase | baseline | ch4_bins | 40_53.8513 | 97 | 40-53.8513 | -1.300838 | 4.327713 | 22.880541 | no |
| ridge_all_modalities | ridge_alr_ch4_all_modalities | per_phase | baseline | ch4_bins | 53.8513_67.7026 | 192 | 53.8513-67.7026 | -0.143509 | 3.416261 | 9.223648 | no |
| ridge_all_modalities | ridge_alr_ch4_all_modalities | per_phase | baseline | ch4_bins | 67.7026_81.5539 | 219 | 67.7026-81.5539 | -0.004628 | -1.215909 | 0.066297 | no |
| ridge_all_modalities | ridge_alr_ch4_all_modalities | per_phase | baseline | ch4_bins | 81.5539_95.4052 | 92 | 81.5539-95.4052 | 1.059368 | -4.173040 | -0.309818 | yes |
| ridge_all_modalities | ridge_alr_ch4_all_modalities | per_phase | exposure | n2_bins | 0.025525_5.00919 | 154 | 0.025525-5.00919 | 5.671503 | -0.124425 | 0.082482 | no |
| ridge_all_modalities | ridge_alr_ch4_all_modalities | per_phase | exposure | n2_bins | 14.9765_19.9602 | 142 | 14.9765-19.9602 | -16.859002 | 1.984335 | 0.228940 | no |
| ridge_all_modalities | ridge_alr_ch4_all_modalities | per_phase | exposure | n2_bins | 5.00919_9.99285 | 166 | 5.00919-9.99285 | 0.094386 | 0.714893 | 0.116355 | no |
| ridge_all_modalities | ridge_alr_ch4_all_modalities | per_phase | exposure | n2_bins | 9.99285_14.9765 | 138 | 9.99285-14.9765 | -8.620268 | 1.890942 | 0.198523 | no |
| ridge_all_modalities | ridge_alr_ch4_all_modalities | per_phase | exposure | ch4_bins | 40_53.8513 | 97 | 40-53.8513 | -1.571054 | 2.158120 | 2.147677 | no |
| ridge_all_modalities | ridge_alr_ch4_all_modalities | per_phase | exposure | ch4_bins | 53.8513_67.7026 | 192 | 53.8513-67.7026 | -0.515756 | 2.066425 | 1.716477 | no |
| ridge_all_modalities | ridge_alr_ch4_all_modalities | per_phase | exposure | ch4_bins | 67.7026_81.5539 | 219 | 67.7026-81.5539 | -0.172176 | 0.472838 | 0.372664 | no |
| ridge_all_modalities | ridge_alr_ch4_all_modalities | per_phase | exposure | ch4_bins | 81.5539_95.4052 | 92 | 81.5539-95.4052 | 0.307523 | -1.137204 | 0.032885 | no |
| ridge_all_modalities | ridge_alr_ch4_all_modalities | per_phase | recovery | n2_bins | 0.025525_5.00919 | 154 | 0.025525-5.00919 | 6.169695 | -0.139232 | 0.085686 | no |
| ridge_all_modalities | ridge_alr_ch4_all_modalities | per_phase | recovery | n2_bins | 14.9765_19.9602 | 142 | 14.9765-19.9602 | -14.894369 | 1.965111 | 0.201655 | no |
| ridge_all_modalities | ridge_alr_ch4_all_modalities | per_phase | recovery | n2_bins | 5.00919_9.99285 | 166 | 5.00919-9.99285 | -0.273746 | 0.719844 | 0.106962 | no |
| ridge_all_modalities | ridge_alr_ch4_all_modalities | per_phase | recovery | n2_bins | 9.99285_14.9765 | 138 | 9.99285-14.9765 | -11.519611 | 2.283517 | 0.196989 | no |
| ridge_all_modalities | ridge_alr_ch4_all_modalities | per_phase | recovery | ch4_bins | 40_53.8513 | 97 | 40-53.8513 | -1.537600 | 2.064579 | 2.147632 | no |
| ridge_all_modalities | ridge_alr_ch4_all_modalities | per_phase | recovery | ch4_bins | 53.8513_67.7026 | 192 | 53.8513-67.7026 | -0.598135 | 2.012960 | 1.399094 | no |
| ridge_all_modalities | ridge_alr_ch4_all_modalities | per_phase | recovery | ch4_bins | 67.7026_81.5539 | 219 | 67.7026-81.5539 | -0.161655 | 0.769598 | 0.634483 | no |
| ridge_all_modalities | ridge_alr_ch4_all_modalities | per_phase | recovery | ch4_bins | 81.5539_95.4052 | 92 | 81.5539-95.4052 | 0.429365 | -1.001783 | 0.006509 | yes |
| ridge_all_modalities | ridge_alr_ch4_all_modalities | per_phase | steady | n2_bins | 0.025525_5.00919 | 154 | 0.025525-5.00919 | 16.446191 | -0.489020 | 0.206523 | no |
| ridge_all_modalities | ridge_alr_ch4_all_modalities | per_phase | steady | n2_bins | 14.9765_19.9602 | 142 | 14.9765-19.9602 | -19.244615 | 1.609608 | 0.251550 | no |
| ridge_all_modalities | ridge_alr_ch4_all_modalities | per_phase | steady | n2_bins | 5.00919_9.99285 | 166 | 5.00919-9.99285 | 2.705243 | 0.219292 | 0.199041 | no |
| ridge_all_modalities | ridge_alr_ch4_all_modalities | per_phase | steady | n2_bins | 9.99285_14.9765 | 138 | 9.99285-14.9765 | -7.904135 | 1.796223 | 0.315412 | no |
| ridge_all_modalities | ridge_alr_ch4_all_modalities | per_phase | steady | ch4_bins | 40_53.8513 | 97 | 40-53.8513 | -2.385339 | 1.277455 | 1.722845 | no |
| ridge_all_modalities | ridge_alr_ch4_all_modalities | per_phase | steady | ch4_bins | 53.8513_67.7026 | 192 | 53.8513-67.7026 | -0.214999 | 1.682708 | 2.139671 | no |
| ridge_all_modalities | ridge_alr_ch4_all_modalities | per_phase | steady | ch4_bins | 67.7026_81.5539 | 219 | 67.7026-81.5539 | 0.016919 | 0.307936 | 0.249892 | no |
| ridge_all_modalities | ridge_alr_ch4_all_modalities | per_phase | steady | ch4_bins | 81.5539_95.4052 | 92 | 81.5539-95.4052 | 1.430912 | -1.421520 | 0.059627 | no |
| ridge_all_modalities | ridge_alr_ch4_all_modalities | early | 0.25 | n2_bins | 0.025525_5.00919 | 154 | 0.025525-5.00919 | 12.328611 | -0.712058 | 0.221167 | no |
| ridge_all_modalities | ridge_alr_ch4_all_modalities | early | 0.25 | n2_bins | 14.9765_19.9602 | 142 | 14.9765-19.9602 | -13.405497 | 2.703215 | 1.381640 | no |
| ridge_all_modalities | ridge_alr_ch4_all_modalities | early | 0.25 | n2_bins | 5.00919_9.99285 | 166 | 5.00919-9.99285 | 2.617941 | 0.372992 | 0.154826 | no |
| ridge_all_modalities | ridge_alr_ch4_all_modalities | early | 0.25 | n2_bins | 9.99285_14.9765 | 138 | 9.99285-14.9765 | -5.236745 | 1.952527 | 0.857286 | no |
| ridge_all_modalities | ridge_alr_ch4_all_modalities | early | 0.25 | ch4_bins | 40_53.8513 | 97 | 40-53.8513 | -1.300838 | 4.327713 | 22.880541 | no |
| ridge_all_modalities | ridge_alr_ch4_all_modalities | early | 0.25 | ch4_bins | 53.8513_67.7026 | 192 | 53.8513-67.7026 | -0.143509 | 3.416261 | 9.223648 | no |
| ridge_all_modalities | ridge_alr_ch4_all_modalities | early | 0.25 | ch4_bins | 67.7026_81.5539 | 219 | 67.7026-81.5539 | -0.004628 | -1.215909 | 0.066297 | no |
| ridge_all_modalities | ridge_alr_ch4_all_modalities | early | 0.25 | ch4_bins | 81.5539_95.4052 | 92 | 81.5539-95.4052 | 1.059368 | -4.173040 | -0.309818 | yes |
| ridge_all_modalities | ridge_alr_ch4_all_modalities | early | 0.5 | n2_bins | 0.025525_5.00919 | 154 | 0.025525-5.00919 | 8.087740 | 0.087408 | 0.164652 | no |
| ridge_all_modalities | ridge_alr_ch4_all_modalities | early | 0.5 | n2_bins | 14.9765_19.9602 | 142 | 14.9765-19.9602 | -11.837549 | 1.446718 | 0.197403 | no |
| ridge_all_modalities | ridge_alr_ch4_all_modalities | early | 0.5 | n2_bins | 5.00919_9.99285 | 166 | 5.00919-9.99285 | 1.290081 | 0.584207 | 0.134448 | no |
| ridge_all_modalities | ridge_alr_ch4_all_modalities | early | 0.5 | n2_bins | 9.99285_14.9765 | 138 | 9.99285-14.9765 | -7.424374 | 1.697916 | 0.189876 | no |
| ridge_all_modalities | ridge_alr_ch4_all_modalities | early | 0.5 | ch4_bins | 40_53.8513 | 97 | 40-53.8513 | -1.600771 | 1.810905 | 2.065536 | no |
| ridge_all_modalities | ridge_alr_ch4_all_modalities | early | 0.5 | ch4_bins | 53.8513_67.7026 | 192 | 53.8513-67.7026 | -0.221464 | 1.899026 | 1.866608 | no |
| ridge_all_modalities | ridge_alr_ch4_all_modalities | early | 0.5 | ch4_bins | 67.7026_81.5539 | 219 | 67.7026-81.5539 | -0.048243 | 0.475463 | 0.474555 | no |
| ridge_all_modalities | ridge_alr_ch4_all_modalities | early | 0.5 | ch4_bins | 81.5539_95.4052 | 92 | 81.5539-95.4052 | 0.621933 | -1.409656 | 0.022635 | no |
| ridge_all_modalities | ridge_alr_ch4_all_modalities | early | 0.75 | n2_bins | 0.025525_5.00919 | 154 | 0.025525-5.00919 | 11.331266 | -0.405789 | 0.155394 | no |
| ridge_all_modalities | ridge_alr_ch4_all_modalities | early | 0.75 | n2_bins | 14.9765_19.9602 | 142 | 14.9765-19.9602 | -12.294955 | 1.389137 | 0.133775 | no |
| ridge_all_modalities | ridge_alr_ch4_all_modalities | early | 0.75 | n2_bins | 5.00919_9.99285 | 166 | 5.00919-9.99285 | 1.471834 | 0.363776 | 0.177935 | no |
| ridge_all_modalities | ridge_alr_ch4_all_modalities | early | 0.75 | n2_bins | 9.99285_14.9765 | 138 | 9.99285-14.9765 | -6.236003 | 1.604487 | 0.190266 | no |
| ridge_all_modalities | ridge_alr_ch4_all_modalities | early | 0.75 | ch4_bins | 40_53.8513 | 97 | 40-53.8513 | -1.752948 | 1.298119 | 1.735915 | no |
| ridge_all_modalities | ridge_alr_ch4_all_modalities | early | 0.75 | ch4_bins | 53.8513_67.7026 | 192 | 53.8513-67.7026 | -0.115899 | 1.363826 | 1.218447 | no |
| ridge_all_modalities | ridge_alr_ch4_all_modalities | early | 0.75 | ch4_bins | 67.7026_81.5539 | 219 | 67.7026-81.5539 | -0.006301 | 0.441620 | 0.329601 | no |
| ridge_all_modalities | ridge_alr_ch4_all_modalities | early | 0.75 | ch4_bins | 81.5539_95.4052 | 92 | 81.5539-95.4052 | 0.988485 | -1.163258 | 0.068205 | no |
| ridge_all_modalities | ridge_alr_ch4_all_modalities | early | 1.0 | n2_bins | 0.025525_5.00919 | 154 | 0.025525-5.00919 | 11.466262 | -0.320393 | 0.223820 | no |
| ridge_all_modalities | ridge_alr_ch4_all_modalities | early | 1.0 | n2_bins | 14.9765_19.9602 | 142 | 14.9765-19.9602 | -12.318765 | 1.399608 | 0.217441 | no |
| ridge_all_modalities | ridge_alr_ch4_all_modalities | early | 1.0 | n2_bins | 5.00919_9.99285 | 166 | 5.00919-9.99285 | 1.560446 | 0.566495 | 0.283776 | no |
| ridge_all_modalities | ridge_alr_ch4_all_modalities | early | 1.0 | n2_bins | 9.99285_14.9765 | 138 | 9.99285-14.9765 | -6.317847 | 1.686738 | 0.217720 | no |
| ridge_all_modalities | ridge_alr_ch4_all_modalities | early | 1.0 | ch4_bins | 40_53.8513 | 97 | 40-53.8513 | -1.574456 | 1.191932 | 1.368252 | no |
| ridge_all_modalities | ridge_alr_ch4_all_modalities | early | 1.0 | ch4_bins | 53.8513_67.7026 | 192 | 53.8513-67.7026 | -0.133598 | 1.492414 | 1.480695 | no |
| ridge_all_modalities | ridge_alr_ch4_all_modalities | early | 1.0 | ch4_bins | 67.7026_81.5539 | 219 | 67.7026-81.5539 | -0.030820 | 0.649938 | 0.625053 | no |
| ridge_all_modalities | ridge_alr_ch4_all_modalities | early | 1.0 | ch4_bins | 81.5539_95.4052 | 92 | 81.5539-95.4052 | 1.012710 | -1.058153 | 0.086666 | no |
| ridge_all_modalities | ridge_ilr_n2_first_all_modalities | per_phase | baseline | n2_bins | 0.025525_5.00919 | 154 | 0.025525-5.00919 | 12.328615 | -0.712058 | 0.221167 | no |
| ridge_all_modalities | ridge_ilr_n2_first_all_modalities | per_phase | baseline | n2_bins | 14.9765_19.9602 | 142 | 14.9765-19.9602 | -13.405500 | 2.703215 | 1.381640 | no |
| ridge_all_modalities | ridge_ilr_n2_first_all_modalities | per_phase | baseline | n2_bins | 5.00919_9.99285 | 166 | 5.00919-9.99285 | 2.617941 | 0.372992 | 0.154826 | no |
| ridge_all_modalities | ridge_ilr_n2_first_all_modalities | per_phase | baseline | n2_bins | 9.99285_14.9765 | 138 | 9.99285-14.9765 | -5.236747 | 1.952528 | 0.857286 | no |
| ridge_all_modalities | ridge_ilr_n2_first_all_modalities | per_phase | baseline | ch4_bins | 40_53.8513 | 97 | 40-53.8513 | -1.300839 | 4.327713 | 22.880542 | no |
| ridge_all_modalities | ridge_ilr_n2_first_all_modalities | per_phase | baseline | ch4_bins | 53.8513_67.7026 | 192 | 53.8513-67.7026 | -0.143509 | 3.416261 | 9.223650 | no |
| ridge_all_modalities | ridge_ilr_n2_first_all_modalities | per_phase | baseline | ch4_bins | 67.7026_81.5539 | 219 | 67.7026-81.5539 | -0.004628 | -1.215909 | 0.066297 | no |
| ridge_all_modalities | ridge_ilr_n2_first_all_modalities | per_phase | baseline | ch4_bins | 81.5539_95.4052 | 92 | 81.5539-95.4052 | 1.059368 | -4.173040 | -0.309818 | yes |
| ridge_all_modalities | ridge_ilr_n2_first_all_modalities | per_phase | exposure | n2_bins | 0.025525_5.00919 | 154 | 0.025525-5.00919 | 5.671504 | -0.124425 | 0.082482 | no |
| ridge_all_modalities | ridge_ilr_n2_first_all_modalities | per_phase | exposure | n2_bins | 14.9765_19.9602 | 142 | 14.9765-19.9602 | -16.859016 | 1.984335 | 0.228940 | no |
| ridge_all_modalities | ridge_ilr_n2_first_all_modalities | per_phase | exposure | n2_bins | 5.00919_9.99285 | 166 | 5.00919-9.99285 | 0.094386 | 0.714892 | 0.116355 | no |
| ridge_all_modalities | ridge_ilr_n2_first_all_modalities | per_phase | exposure | n2_bins | 9.99285_14.9765 | 138 | 9.99285-14.9765 | -8.620273 | 1.890943 | 0.198523 | no |
| ridge_all_modalities | ridge_ilr_n2_first_all_modalities | per_phase | exposure | ch4_bins | 40_53.8513 | 97 | 40-53.8513 | -1.571055 | 2.158120 | 2.147676 | no |
| ridge_all_modalities | ridge_ilr_n2_first_all_modalities | per_phase | exposure | ch4_bins | 53.8513_67.7026 | 192 | 53.8513-67.7026 | -0.515756 | 2.066425 | 1.716476 | no |
| ridge_all_modalities | ridge_ilr_n2_first_all_modalities | per_phase | exposure | ch4_bins | 67.7026_81.5539 | 219 | 67.7026-81.5539 | -0.172177 | 0.472838 | 0.372664 | no |
| ridge_all_modalities | ridge_ilr_n2_first_all_modalities | per_phase | exposure | ch4_bins | 81.5539_95.4052 | 92 | 81.5539-95.4052 | 0.307523 | -1.137204 | 0.032885 | no |
| ridge_all_modalities | ridge_ilr_n2_first_all_modalities | per_phase | recovery | n2_bins | 0.025525_5.00919 | 154 | 0.025525-5.00919 | 6.169697 | -0.139231 | 0.085687 | no |
| ridge_all_modalities | ridge_ilr_n2_first_all_modalities | per_phase | recovery | n2_bins | 14.9765_19.9602 | 142 | 14.9765-19.9602 | -14.894393 | 1.965113 | 0.201656 | no |
| ridge_all_modalities | ridge_ilr_n2_first_all_modalities | per_phase | recovery | n2_bins | 5.00919_9.99285 | 166 | 5.00919-9.99285 | -0.273746 | 0.719844 | 0.106962 | no |
| ridge_all_modalities | ridge_ilr_n2_first_all_modalities | per_phase | recovery | n2_bins | 9.99285_14.9765 | 138 | 9.99285-14.9765 | -11.519619 | 2.283519 | 0.196990 | no |
| ridge_all_modalities | ridge_ilr_n2_first_all_modalities | per_phase | recovery | ch4_bins | 40_53.8513 | 97 | 40-53.8513 | -1.537601 | 2.064581 | 2.147632 | no |
| ridge_all_modalities | ridge_ilr_n2_first_all_modalities | per_phase | recovery | ch4_bins | 53.8513_67.7026 | 192 | 53.8513-67.7026 | -0.598136 | 2.012962 | 1.399096 | no |
| ridge_all_modalities | ridge_ilr_n2_first_all_modalities | per_phase | recovery | ch4_bins | 67.7026_81.5539 | 219 | 67.7026-81.5539 | -0.161655 | 0.769599 | 0.634484 | no |
| ridge_all_modalities | ridge_ilr_n2_first_all_modalities | per_phase | recovery | ch4_bins | 81.5539_95.4052 | 92 | 81.5539-95.4052 | 0.429365 | -1.001784 | 0.006509 | yes |
| ridge_all_modalities | ridge_ilr_n2_first_all_modalities | per_phase | steady | n2_bins | 0.025525_5.00919 | 154 | 0.025525-5.00919 | 16.446193 | -0.489021 | 0.206524 | no |
| ridge_all_modalities | ridge_ilr_n2_first_all_modalities | per_phase | steady | n2_bins | 14.9765_19.9602 | 142 | 14.9765-19.9602 | -19.244618 | 1.609609 | 0.251550 | no |
| ridge_all_modalities | ridge_ilr_n2_first_all_modalities | per_phase | steady | n2_bins | 5.00919_9.99285 | 166 | 5.00919-9.99285 | 2.705243 | 0.219292 | 0.199041 | no |
| ridge_all_modalities | ridge_ilr_n2_first_all_modalities | per_phase | steady | n2_bins | 9.99285_14.9765 | 138 | 9.99285-14.9765 | -7.904137 | 1.796223 | 0.315412 | no |
| ridge_all_modalities | ridge_ilr_n2_first_all_modalities | per_phase | steady | ch4_bins | 40_53.8513 | 97 | 40-53.8513 | -2.385339 | 1.277455 | 1.722845 | no |
| ridge_all_modalities | ridge_ilr_n2_first_all_modalities | per_phase | steady | ch4_bins | 53.8513_67.7026 | 192 | 53.8513-67.7026 | -0.215000 | 1.682708 | 2.139671 | no |
| ridge_all_modalities | ridge_ilr_n2_first_all_modalities | per_phase | steady | ch4_bins | 67.7026_81.5539 | 219 | 67.7026-81.5539 | 0.016919 | 0.307936 | 0.249892 | no |
| ridge_all_modalities | ridge_ilr_n2_first_all_modalities | per_phase | steady | ch4_bins | 81.5539_95.4052 | 92 | 81.5539-95.4052 | 1.430912 | -1.421520 | 0.059627 | no |
| ridge_all_modalities | ridge_ilr_n2_first_all_modalities | early | 0.25 | n2_bins | 0.025525_5.00919 | 154 | 0.025525-5.00919 | 12.328615 | -0.712058 | 0.221167 | no |
| ridge_all_modalities | ridge_ilr_n2_first_all_modalities | early | 0.25 | n2_bins | 14.9765_19.9602 | 142 | 14.9765-19.9602 | -13.405500 | 2.703215 | 1.381640 | no |
| ridge_all_modalities | ridge_ilr_n2_first_all_modalities | early | 0.25 | n2_bins | 5.00919_9.99285 | 166 | 5.00919-9.99285 | 2.617941 | 0.372992 | 0.154826 | no |
| ridge_all_modalities | ridge_ilr_n2_first_all_modalities | early | 0.25 | n2_bins | 9.99285_14.9765 | 138 | 9.99285-14.9765 | -5.236747 | 1.952528 | 0.857286 | no |
| ridge_all_modalities | ridge_ilr_n2_first_all_modalities | early | 0.25 | ch4_bins | 40_53.8513 | 97 | 40-53.8513 | -1.300839 | 4.327713 | 22.880542 | no |
| ridge_all_modalities | ridge_ilr_n2_first_all_modalities | early | 0.25 | ch4_bins | 53.8513_67.7026 | 192 | 53.8513-67.7026 | -0.143509 | 3.416261 | 9.223650 | no |
| ridge_all_modalities | ridge_ilr_n2_first_all_modalities | early | 0.25 | ch4_bins | 67.7026_81.5539 | 219 | 67.7026-81.5539 | -0.004628 | -1.215909 | 0.066297 | no |
| ridge_all_modalities | ridge_ilr_n2_first_all_modalities | early | 0.25 | ch4_bins | 81.5539_95.4052 | 92 | 81.5539-95.4052 | 1.059368 | -4.173040 | -0.309818 | yes |
| ridge_all_modalities | ridge_ilr_n2_first_all_modalities | early | 0.5 | n2_bins | 0.025525_5.00919 | 154 | 0.025525-5.00919 | 8.087741 | 0.087409 | 0.164653 | no |
| ridge_all_modalities | ridge_ilr_n2_first_all_modalities | early | 0.5 | n2_bins | 14.9765_19.9602 | 142 | 14.9765-19.9602 | -11.837548 | 1.446719 | 0.197403 | no |
| ridge_all_modalities | ridge_ilr_n2_first_all_modalities | early | 0.5 | n2_bins | 5.00919_9.99285 | 166 | 5.00919-9.99285 | 1.290081 | 0.584209 | 0.134449 | no |
| ridge_all_modalities | ridge_ilr_n2_first_all_modalities | early | 0.5 | n2_bins | 9.99285_14.9765 | 138 | 9.99285-14.9765 | -7.424373 | 1.697918 | 0.189876 | no |
| ridge_all_modalities | ridge_ilr_n2_first_all_modalities | early | 0.5 | ch4_bins | 40_53.8513 | 97 | 40-53.8513 | -1.600771 | 1.810908 | 2.065541 | no |
| ridge_all_modalities | ridge_ilr_n2_first_all_modalities | early | 0.5 | ch4_bins | 53.8513_67.7026 | 192 | 53.8513-67.7026 | -0.221464 | 1.899028 | 1.866611 | no |
| ridge_all_modalities | ridge_ilr_n2_first_all_modalities | early | 0.5 | ch4_bins | 67.7026_81.5539 | 219 | 67.7026-81.5539 | -0.048243 | 0.475464 | 0.474556 | no |
| ridge_all_modalities | ridge_ilr_n2_first_all_modalities | early | 0.5 | ch4_bins | 81.5539_95.4052 | 92 | 81.5539-95.4052 | 0.621933 | -1.409657 | 0.022635 | no |
| ridge_all_modalities | ridge_ilr_n2_first_all_modalities | early | 0.75 | n2_bins | 0.025525_5.00919 | 154 | 0.025525-5.00919 | 11.331267 | -0.405789 | 0.155394 | no |
| ridge_all_modalities | ridge_ilr_n2_first_all_modalities | early | 0.75 | n2_bins | 14.9765_19.9602 | 142 | 14.9765-19.9602 | -12.294962 | 1.389138 | 0.133775 | no |
| ridge_all_modalities | ridge_ilr_n2_first_all_modalities | early | 0.75 | n2_bins | 5.00919_9.99285 | 166 | 5.00919-9.99285 | 1.471834 | 0.363776 | 0.177935 | no |
| ridge_all_modalities | ridge_ilr_n2_first_all_modalities | early | 0.75 | n2_bins | 9.99285_14.9765 | 138 | 9.99285-14.9765 | -6.236005 | 1.604488 | 0.190266 | no |
| ridge_all_modalities | ridge_ilr_n2_first_all_modalities | early | 0.75 | ch4_bins | 40_53.8513 | 97 | 40-53.8513 | -1.752949 | 1.298119 | 1.735915 | no |
| ridge_all_modalities | ridge_ilr_n2_first_all_modalities | early | 0.75 | ch4_bins | 53.8513_67.7026 | 192 | 53.8513-67.7026 | -0.115900 | 1.363826 | 1.218447 | no |
| ridge_all_modalities | ridge_ilr_n2_first_all_modalities | early | 0.75 | ch4_bins | 67.7026_81.5539 | 219 | 67.7026-81.5539 | -0.006301 | 0.441620 | 0.329602 | no |
| ridge_all_modalities | ridge_ilr_n2_first_all_modalities | early | 0.75 | ch4_bins | 81.5539_95.4052 | 92 | 81.5539-95.4052 | 0.988485 | -1.163258 | 0.068205 | no |
| ridge_all_modalities | ridge_ilr_n2_first_all_modalities | early | 1.0 | n2_bins | 0.025525_5.00919 | 154 | 0.025525-5.00919 | 11.466266 | -0.320394 | 0.223820 | no |
| ridge_all_modalities | ridge_ilr_n2_first_all_modalities | early | 1.0 | n2_bins | 14.9765_19.9602 | 142 | 14.9765-19.9602 | -12.318769 | 1.399609 | 0.217441 | no |
| ridge_all_modalities | ridge_ilr_n2_first_all_modalities | early | 1.0 | n2_bins | 5.00919_9.99285 | 166 | 5.00919-9.99285 | 1.560446 | 0.566495 | 0.283776 | no |
| ridge_all_modalities | ridge_ilr_n2_first_all_modalities | early | 1.0 | n2_bins | 9.99285_14.9765 | 138 | 9.99285-14.9765 | -6.317847 | 1.686738 | 0.217720 | no |
| ridge_all_modalities | ridge_ilr_n2_first_all_modalities | early | 1.0 | ch4_bins | 40_53.8513 | 97 | 40-53.8513 | -1.574456 | 1.191932 | 1.368251 | no |
| ridge_all_modalities | ridge_ilr_n2_first_all_modalities | early | 1.0 | ch4_bins | 53.8513_67.7026 | 192 | 53.8513-67.7026 | -0.133598 | 1.492414 | 1.480696 | no |
| ridge_all_modalities | ridge_ilr_n2_first_all_modalities | early | 1.0 | ch4_bins | 67.7026_81.5539 | 219 | 67.7026-81.5539 | -0.030820 | 0.649938 | 0.625053 | no |
| ridge_all_modalities | ridge_ilr_n2_first_all_modalities | early | 1.0 | ch4_bins | 81.5539_95.4052 | 92 | 81.5539-95.4052 | 1.012711 | -1.058154 | 0.086666 | no |
