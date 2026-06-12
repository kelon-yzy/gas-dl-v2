# Composition Label Inspection

- dataset_dir: `data/wv4-formal-hitran-standard-6000`
- alr_reference: `x_CH4`
- recommended epsilon: `4.20499985e-06` (train_min_positive_half)

| split | rows | component | zero count | zero ratio | min % | min positive % | max % |
|---|---:|---|---:|---:|---:|---:|---:|
| train | 4200 | x_H2 | 0 | 0.000000 | 0.000841 | 0.000841 | 29.9982 |
| train | 4200 | x_CH4 | 0 | 0.000000 | 40 | 40 | 98.0956 |
| train | 4200 | x_CO2 | 0 | 0.000000 | 0.001489 | 0.001489 | 14.9925 |
| train | 4200 | x_N2 | 0 | 0.000000 | 0.00156 | 0.00156 | 19.9976 |
| val | 900 | x_H2 | 0 | 0.000000 | 0.013113 | 0.013113 | 29.9236 |
| val | 900 | x_CH4 | 0 | 0.000000 | 40 | 40 | 98.1319 |
| val | 900 | x_CO2 | 0 | 0.000000 | 0.005556 | 0.005556 | 14.9906 |
| val | 900 | x_N2 | 0 | 0.000000 | 0.017576 | 0.017576 | 19.9866 |
| test | 600 | x_H2 | 0 | 0.000000 | 0.017488 | 0.017488 | 29.9797 |
| test | 600 | x_CH4 | 0 | 0.000000 | 40 | 40 | 95.4052 |
| test | 600 | x_CO2 | 0 | 0.000000 | 0.01556 | 0.01556 | 14.9983 |
| test | 600 | x_N2 | 0 | 0.000000 | 0.025525 | 0.025525 | 19.9602 |
| extrapolation | 300 | x_H2 | 0 | 0.000000 | 0.035294 | 0.035294 | 29.8484 |
| extrapolation | 300 | x_CH4 | 0 | 0.000000 | 40.2558 | 40.2558 | 95.3983 |
| extrapolation | 300 | x_CO2 | 0 | 0.000000 | 0.064844 | 0.064844 | 14.9957 |
| extrapolation | 300 | x_N2 | 0 | 0.000000 | 0.044362 | 0.044362 | 19.9933 |

| split | ALR reference | log-reference mean | log-reference variance |
|---|---|---:|---:|
| train | x_CH4 | -0.407871 | 0.035208 |
| val | x_CH4 | -0.408936 | 0.035498 |
| test | x_CH4 | -0.405640 | 0.037285 |
| extrapolation | x_CH4 | -0.412989 | 0.034001 |
