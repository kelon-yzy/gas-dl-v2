# S1 3 × 3 Fisher/CRB/angle 冻结表

> source: `gas_information_bench/configs/p2_s1_grid.json`
> source_sha256: `D193585A12932C78A8383A73058AFB989318DAB83EAE4953E4C4B74F1C682310`

完整 `effective_fisher` 3 × 3 矩阵保留在源 JSON；本表报告其 trace、四组分 CRB P90、最大容差比、实际主夹角和条件数。

| cell | 信息档 | 共线档 | trace(F_eff) | CRB P90 N2/CO2/O2/Ar | max(CRB P90/tau) | angle (deg) | cond(F_eff) | accessible |
|---|---|---|---:|---|---:|---:|---:|---|
| GIB-S1-SUF-HIG | sufficient | high_collinearity | 45744.124718 | 0.032542/0.013079/0.038281/0.019763 | 0.435962 | 9.962666 | 29.731558 | true |
| GIB-S1-SUF-MED | sufficient | medium_collinearity | 146424.142645 | 0.026128/0.009755/0.035665/0.017036 | 0.356647 | 44.950440 | 83.053657 | true |
| GIB-S1-SUF-LOW | sufficient | low_collinearity | 232560.597700 | 0.025593/0.009569/0.034840/0.015376 | 0.348404 | 79.995807 | 136.163277 | true |
| GIB-S1-CRI-HIG | critical | high_collinearity | 8497.009444 | 0.087844/0.032460/0.106934/0.050106 | 1.098051 | 9.962666 | 37.346725 | true |
| GIB-S1-CRI-MED | critical | medium_collinearity | 19535.242318 | 0.074434/0.027660/0.100893/0.046767 | 1.008931 | 44.950440 | 89.931998 | true |
| GIB-S1-CRI-LOW | critical | low_collinearity | 28628.422432 | 0.073109/0.027334/0.099447/0.043789 | 0.994472 | 79.995807 | 136.677730 | true |
| GIB-S1-INS-HIG | insufficient | high_collinearity | 2545.176599 | 0.183218/0.067109/0.226945/0.105821 | 2.290226 | 9.962666 | 55.774315 | true |
| GIB-S1-INS-MED | insufficient | medium_collinearity | 4325.546409 | 0.163090/0.060580/0.219787/0.098897 | 2.197871 | 44.950440 | 95.401223 | true |
| GIB-S1-INS-LOW | insufficient | low_collinearity | 5934.623833 | 0.160817/0.060128/0.218707/0.096231 | 2.187075 | 79.995807 | 137.110074 | true |

该表只复算冻结产物，不修改门值，不代表已生成 pilot 数据。
