This repository represents my master thesis project. The data preprocessing and organizing is based upon the work of Harutyunyan et al. (2019) and their Github repository: https://github.com/YerevaNN/mimic3-benchmarks

It is updated for the MIMIC-IV database version 1.0 (Johnson et al. 2020), reorganized and optimized, e.g. by imbalancedness handling.

The task is to tranfer learnings from the whole database (source dataset) to a rare disease patient cohort (ARDS, target dataset) via domain adaptation by pretraining a model on the source dataset and fine-tuning it on the target dataset. The outcome of interest is in-hospital-mortality.
<br>
<br>

Johnson, A., Bulgarelli, L., Pollard, T., Horng, S., Celi, L. A., & Mark, R. (2020). Mimic-iv. version 1.0). PhysioNet. https://doi. org/10.13026/a3wn-hq05.

Harutyunyan, H., Khachatrian, H., Kale, D. C., Ver Steeg, G., & Galstyan, A. (2019). Multitask learning and benchmarking with clinical time series data. Scientific data, 6(1), 1-18.
