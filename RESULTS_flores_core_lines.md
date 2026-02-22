# Tokenizer Evaluation Results — flores_core / lines

_Last updated: 2026-02-22 23:19:59_

| Tokenizer | Vocab Size | Fertility ↓ | Compression Rate ↑ | Vocab Util. ↑ | TTR ↑ | Shannon Entropy ↑ | Avg Token Rank ↓ | Gini ↓ | 3-Digit Align. F1 ↑ | Op. Isolation ↑ | Languages | Dataset | User | Date |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| aggresssive_N2_160000 [19k] | 19,051 | 62.586 | 0.019 | 0.295 | 0.0069 | 9.53 | 382.8 | 0.120 | 0.635 | 0.382 | 13 | flores_core | vkanjira | 2026-02-22 |
| Boundless pairwise aggressive n2 (dkletz, flores_core) | 5,033 | 75.587 | 0.017 | **0.480** | 0.0025 | 7.88 | **121.7** | 0.131 | --- | 0.087 | 13 | flores_core | dkletz | 2026-02-22 |
| Classical (meistecl, flores_core) | 128,004 | 36.127 | 0.031 | 0.369 | 0.1009 | 12.91 | 6524.2 | 0.049 | **0.779** | 0.190 | 13 | flores_core | meistecl | 2026-02-21 |
| Parity-aware (meistecl, flores_core) | 128,004 | 38.139 | 0.029 | 0.322 | 0.0835 | 12.73 | 5367.5 | **0.012** | 0.742 | 0.191 | 13 | flores_core | meistecl | 2026-02-21 |
| Parity-aware (hybrid) (meistecl, flores_core) | 128,004 | 37.385 | 0.030 | 0.330 | 0.0872 | 12.80 | 5693.5 | 0.028 | 0.753 | 0.190 | 13 | flores_core | meistecl | 2026-02-21 |
| Gemma 3 (262k) (meistecl, flores_core) | 262,145 | 34.075 | 0.033 | 0.217 | 0.1286 | 13.05 | 7779.7 | 0.054 | 0.446 | 0.346 | 13 | flores_core | meistecl | 2026-02-21 |
| Qwen 3 (151k) (meistecl, flores_core) | 151,669 | 44.867 | 0.029 | 0.228 | 0.0594 | 11.94 | 3303.4 | 0.181 | 0.379 | **0.527** | 13 | flores_core | meistecl | 2026-02-21 |
| Mistral-Nemo (meistecl, flores_core) | 131,072 | 37.419 | 0.031 | 0.341 | 0.0921 | 12.84 | 5610.9 | 0.069 | 0.379 | 0.139 | 13 | flores_core | meistecl | 2026-02-21 |
| DeepSeek R1 (meistecl, flores_core) | 128,815 | 41.449 | 0.029 | 0.285 | 0.0684 | 12.41 | 3944.5 | 0.136 | 0.779 | 0.188 | 13 | flores_core | meistecl | 2026-02-21 |
| Kimi (160k) (meistecl, flores_core) | 163,840 | 45.112 | 0.027 | 0.173 | 0.0486 | 12.16 | 2588.9 | 0.137 | 0.779 | 0.489 | 13 | flores_core | meistecl | 2026-02-21 |
| GLM (151k) (meistecl, flores_core) | 151,343 | 46.472 | 0.027 | 0.251 | 0.0630 | 11.71 | 3402.5 | 0.197 | 0.260 | 0.000 | 13 | flores_core | meistecl | 2026-02-21 |
| GPT OSS (201k) (meistecl, flores_core) | 200,019 | 34.730 | 0.033 | 0.245 | 0.1087 | **13.22** | 6659.5 | 0.065 | 0.779 | 0.354 | 13 | flores_core | meistecl | 2026-02-21 |
| Apertus (meistecl, flores_core) | 131,072 | 37.419 | 0.031 | 0.341 | 0.0921 | 12.84 | 5610.9 | 0.069 | 0.379 | 0.139 | 13 | flores_core | meistecl | 2026-02-21 |
| Apertus_Coarse_Family (ayavuz, flores_core) | --- | 40.070 | 0.029 | 0.340 | 0.0858 | 12.49 | --- | 0.103 | --- | --- | 13 | flores_core | ayavuz | 2026-02-21 |
| Apertus_Fine_Family (ayavuz, flores_core) | --- | 40.619 | 0.029 | 0.336 | 0.0837 | 12.43 | --- | 0.114 | --- | --- | 13 | flores_core | ayavuz | 2026-02-21 |
| Apertus_Per_Lang (ayavuz, flores_core) | --- | 38.592 | 0.030 | 0.333 | 0.0872 | 12.62 | --- | 0.083 | --- | --- | 13 | flores_core | ayavuz | 2026-02-21 |
| BPE_Coarse_Family (ayavuz, flores_core) | --- | 42.935 | 0.027 | 0.245 | 0.0577 | 12.46 | --- | 0.083 | --- | --- | 13 | flores_core | ayavuz | 2026-02-21 |
| BPE_Fine_Family (ayavuz, flores_core) | --- | 44.451 | 0.026 | 0.237 | 0.0539 | 12.32 | --- | 0.082 | --- | --- | 13 | flores_core | ayavuz | 2026-02-21 |
| BPE_Per_Lang (ayavuz, flores_core) | --- | 42.585 | 0.027 | 0.228 | 0.0541 | 12.43 | --- | 0.082 | --- | --- | 13 | flores_core | ayavuz | 2026-02-21 |
| Unigram_Coarse_Family (ayavuz, flores_core) | --- | 47.868 | 0.024 | 0.254 | 0.0536 | 11.73 | --- | 0.066 | --- | --- | 13 | flores_core | ayavuz | 2026-02-21 |
| Unigram_Fine_Family (ayavuz, flores_core) | --- | 47.843 | 0.024 | 0.247 | 0.0521 | 11.72 | --- | 0.065 | --- | --- | 13 | flores_core | ayavuz | 2026-02-21 |
| Unigram_Per_Lang (ayavuz, flores_core) | --- | 47.415 | 0.024 | 0.233 | 0.0497 | 11.70 | --- | 0.061 | --- | --- | 13 | flores_core | ayavuz | 2026-02-21 |
| Zip2Zip/Gemma 3 (512 codebook) (saibo, flores_core) | 262,657 | **33.668** | **0.034** | 0.217 | **0.1303** | 13.08 | 7808.0 | 0.053 | --- | --- | 13 | flores_core | saibo | 2026-02-18 |
| Zip2Zip/Qwen 3 (512 codebook) (saibo, flores_core) | 152,181 | 43.280 | 0.029 | 0.228 | 0.0619 | 12.15 | 3441.9 | 0.164 | --- | --- | 13 | flores_core | saibo | 2026-02-18 |
| Zip2Zip/Mistral-Nemo (512 codebook) (saibo, flores_core) | 131,584 | 36.910 | 0.031 | 0.340 | 0.0936 | 12.86 | 5639.2 | 0.068 | --- | --- | 13 | flores_core | saibo | 2026-02-18 |
| Zip2Zip/DeepSeek R1 (512 codebook) (saibo, flores_core) | 129,327 | 40.814 | 0.029 | 0.285 | 0.0697 | 12.44 | 3985.5 | 0.133 | --- | --- | 13 | flores_core | saibo | 2026-02-18 |
| Zip2Zip/GPT OSS (512 codebook) (saibo, flores_core) | 200,531 | 34.314 | 0.033 | 0.244 | 0.1102 | 13.22 | 6676.6 | 0.064 | --- | --- | 13 | flores_core | saibo | 2026-02-18 |
| Zip2Zip/Apertus (512 codebook) (saibo, flores_core) | 131,584 | 36.910 | 0.031 | 0.340 | 0.0936 | 12.86 | 5639.2 | 0.068 | --- | --- | 13 | flores_core | saibo | 2026-02-18 |
