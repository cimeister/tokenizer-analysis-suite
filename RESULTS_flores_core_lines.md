# Tokenizer Evaluation Results

_Last updated: 2026-02-23 21:56:13_

| Tokenizer | Vocab Size | Fertility ↓ | Compression Rate ↑ | Vocab Util. ↑ | Avg Token Rank ↓ | Gini ↓ | 3-Digit Align. F1 ↑ | Op. Isolation ↑ | AST Align. ↑ | Ident. Frag. ↓ | Indent Cons. ↑ | Languages | TTR | Shannon Entropy | Dataset | User | Date |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Classical (meistecl, flores_core) | 128,004 | 36.127 | 0.031 | 0.369 | 6524.2 | 0.049 | **0.779** | 0.190 | 0.149 | 0.055 | **1.000** | 13 | 0.1009 | 12.91 | flores_core | meistecl | 2026-02-23 |
| Parity-aware (meistecl, flores_core) | 128,004 | 38.139 | 0.029 | 0.322 | 5367.5 | **0.012** | 0.742 | 0.191 | 0.195 | 0.083 | 1.000 | 13 | 0.0835 | 12.73 | flores_core | meistecl | 2026-02-23 |
| Parity-aware (hybrid) (meistecl, flores_core) | 128,004 | 37.385 | 0.030 | 0.330 | 5693.5 | 0.028 | 0.753 | 0.190 | 0.160 | 0.066 | 1.000 | 13 | 0.0872 | 12.80 | flores_core | meistecl | 2026-02-23 |
| Gemma 3 (262k) (meistecl, flores_core) | 262,145 | 34.075 | 0.033 | 0.217 | 7779.7 | 0.054 | 0.446 | 0.346 | 0.094 | 0.032 | 0.977 | 13 | 0.1286 | 13.05 | flores_core | meistecl | 2026-02-23 |
| Qwen 3 (151k) (meistecl, flores_core) | 151,669 | 44.867 | 0.029 | 0.228 | 3303.4 | 0.181 | 0.379 | 0.527 | 0.602 | 0.381 | 1.000 | 13 | 0.0594 | 11.94 | flores_core | meistecl | 2026-02-23 |
| Mistral-Nemo (meistecl, flores_core) | 131,072 | 37.419 | 0.031 | 0.341 | 5610.9 | 0.069 | 0.379 | 0.139 | 0.063 | 0.036 | 1.000 | 13 | 0.0921 | 12.84 | flores_core | meistecl | 2026-02-23 |
| DeepSeek R1 (meistecl, flores_core) | 128,815 | 41.449 | 0.029 | 0.285 | 3944.5 | 0.136 | 0.779 | 0.188 | 0.000 | --- | 1.000 | 13 | 0.0684 | 12.41 | flores_core | meistecl | 2026-02-23 |
| Kimi (160k) (meistecl, flores_core) | 163,840 | 45.112 | 0.027 | 0.173 | 2588.9 | 0.137 | 0.779 | 0.489 | **0.607** | 0.406 | 1.000 | 13 | 0.0486 | 12.16 | flores_core | meistecl | 2026-02-23 |
| GLM (151k) (meistecl, flores_core) | 151,343 | 46.472 | 0.027 | 0.251 | 3402.5 | 0.197 | 0.270 | 0.000 | 0.001 | **0.000** | 1.000 | 13 | 0.0630 | 11.71 | flores_core | meistecl | 2026-02-23 |
| GPT OSS (201k) (meistecl, flores_core) | 200,019 | 34.730 | 0.033 | 0.245 | 6659.5 | 0.065 | 0.779 | 0.354 | 0.606 | 0.400 | 1.000 | 13 | 0.1087 | 13.22 | flores_core | meistecl | 2026-02-23 |
| Apertus (meistecl, flores_core) | 131,072 | 37.419 | 0.031 | 0.341 | 5610.9 | 0.069 | 0.379 | 0.139 | 0.063 | 0.036 | 1.000 | 13 | 0.0921 | 12.84 | flores_core | meistecl | 2026-02-23 |
| Apertus_Coarse_Family (ayavuz, flores_core) | --- | 40.070 | 0.029 | 0.340 | --- | 0.103 | --- | --- | --- | --- | --- | 13 | 0.0858 | 12.49 | flores_core | ayavuz | 2026-02-21 |
| Apertus_Fine_Family (ayavuz, flores_core) | --- | 40.619 | 0.029 | 0.336 | --- | 0.114 | --- | --- | --- | --- | --- | 13 | 0.0837 | 12.43 | flores_core | ayavuz | 2026-02-21 |
| Apertus_Per_Lang (ayavuz, flores_core) | --- | 38.592 | 0.030 | 0.333 | --- | 0.083 | --- | --- | --- | --- | --- | 13 | 0.0872 | 12.62 | flores_core | ayavuz | 2026-02-21 |
| BPE_Coarse_Family (ayavuz, flores_core) | --- | 42.935 | 0.027 | 0.245 | --- | 0.083 | --- | --- | --- | --- | --- | 13 | 0.0577 | 12.46 | flores_core | ayavuz | 2026-02-21 |
| BPE_Fine_Family (ayavuz, flores_core) | --- | 44.451 | 0.026 | 0.237 | --- | 0.082 | --- | --- | --- | --- | --- | 13 | 0.0539 | 12.32 | flores_core | ayavuz | 2026-02-21 |
| BPE_Per_Lang (ayavuz, flores_core) | --- | 42.585 | 0.027 | 0.228 | --- | 0.082 | --- | --- | --- | --- | --- | 13 | 0.0541 | 12.43 | flores_core | ayavuz | 2026-02-21 |
| Unigram_Coarse_Family (ayavuz, flores_core) | --- | 47.868 | 0.024 | 0.254 | --- | 0.066 | --- | --- | --- | --- | --- | 13 | 0.0536 | 11.73 | flores_core | ayavuz | 2026-02-21 |
| Unigram_Fine_Family (ayavuz, flores_core) | --- | 47.843 | 0.024 | 0.247 | --- | 0.065 | --- | --- | --- | --- | --- | 13 | 0.0521 | 11.72 | flores_core | ayavuz | 2026-02-21 |
| Unigram_Per_Lang (ayavuz, flores_core) | --- | 47.415 | 0.024 | 0.233 | --- | 0.061 | --- | --- | --- | --- | --- | 13 | 0.0497 | 11.70 | flores_core | ayavuz | 2026-02-21 |
| Zip2Zip/Gemma 3 (512 codebook) (saibo, flores_core) | 262,657 | **33.668** | **0.034** | 0.217 | 7808.0 | 0.053 | --- | --- | --- | --- | --- | 13 | 0.1303 | 13.08 | flores_core | saibo | 2026-02-18 |
| Zip2Zip/Qwen 3 (512 codebook) (saibo, flores_core) | 152,181 | 43.280 | 0.029 | 0.228 | 3441.9 | 0.164 | --- | --- | --- | --- | --- | 13 | 0.0619 | 12.15 | flores_core | saibo | 2026-02-18 |
| Zip2Zip/Mistral-Nemo (512 codebook) (saibo, flores_core) | 131,584 | 36.910 | 0.031 | 0.340 | 5639.2 | 0.068 | --- | --- | --- | --- | --- | 13 | 0.0936 | 12.86 | flores_core | saibo | 2026-02-18 |
| Zip2Zip/DeepSeek R1 (512 codebook) (saibo, flores_core) | 129,327 | 40.814 | 0.029 | 0.285 | 3985.5 | 0.133 | --- | --- | --- | --- | --- | 13 | 0.0697 | 12.44 | flores_core | saibo | 2026-02-18 |
| Zip2Zip/GPT OSS (512 codebook) (saibo, flores_core) | 200,531 | 34.314 | 0.033 | 0.244 | 6676.6 | 0.064 | --- | --- | --- | --- | --- | 13 | 0.1102 | 13.22 | flores_core | saibo | 2026-02-18 |
| Zip2Zip/Apertus (512 codebook) (saibo, flores_core) | 131,584 | 36.910 | 0.031 | 0.340 | 5639.2 | 0.068 | --- | --- | --- | --- | --- | 13 | 0.0936 | 12.86 | flores_core | saibo | 2026-02-18 |
| Apertus (cmeister747, flores_core) | 131,072 | 37.419 | 0.031 | 0.341 | 5610.9 | 0.069 | 0.379 | 0.139 | --- | --- | --- | 13 | 0.0921 | 12.84 | flores_core | cmeister747 | 2026-02-23 |
| Parity-aware (hybrid) (cmeister747, flores_core) | 128,004 | 37.385 | 0.030 | 0.330 | 5693.5 | 0.028 | 0.753 | 0.190 | --- | --- | --- | 13 | 0.0872 | 12.80 | flores_core | cmeister747 | 2026-02-23 |
| SuperBPE a (cmeister747, flores_core) | 130,000 | 40.307 | 0.028 | 0.241 | 4421.4 | 0.069 | 0.779 | 0.379 | --- | --- | --- | 13 | 0.0601 | 13.05 | flores_core | cmeister747 | 2026-02-23 |
| SuperBPE b (cmeister747, flores_core) | 130,000 | 42.557 | 0.028 | 0.253 | 4104.3 | 0.122 | 0.779 | 0.966 | --- | --- | --- | 13 | 0.0597 | 12.55 | flores_core | cmeister747 | 2026-02-23 |
| SuperBPE qwen regex (cmeister747, flores_core) | 130,000 | 42.575 | 0.027 | 0.280 | 5048.7 | 0.059 | 0.379 | 0.312 | --- | --- | --- | 13 | 0.0659 | 13.27 | flores_core | cmeister747 | 2026-02-23 |
| SuperBPE PA base (cmeister747, flores_core) | 130,000 | 43.548 | 0.027 | 0.236 | 3785.0 | 0.118 | 0.779 | **0.967** | --- | --- | --- | 13 | 0.0545 | 12.48 | flores_core | cmeister747 | 2026-02-23 |
| PA BPE grouped (cmeister747, flores_core) | 120,260 | 45.497 | 0.025 | 0.199 | 2898.5 | 0.073 | 0.712 | 0.191 | --- | --- | --- | 13 | 0.0406 | 12.19 | flores_core | cmeister747 | 2026-02-23 |
| Qwen 3 (151k) (cmeister747, flores_core) | 151,669 | 44.867 | 0.029 | 0.228 | 3303.4 | 0.181 | 0.379 | 0.527 | --- | --- | --- | 13 | 0.0594 | 11.94 | flores_core | cmeister747 | 2026-02-23 |
| DeepSeek R1 (cmeister747, flores_core) | 128,815 | 40.449 | 0.030 | 0.285 | 4041.0 | 0.140 | 0.779 | 0.504 | --- | --- | --- | 13 | 0.0701 | 12.55 | flores_core | cmeister747 | 2026-02-23 |
| Kimi (160k) (cmeister747, flores_core) | 163,840 | 45.112 | 0.027 | 0.173 | 2588.9 | 0.137 | 0.779 | 0.489 | --- | --- | --- | 13 | 0.0486 | 12.16 | flores_core | cmeister747 | 2026-02-23 |
| GLM (151k) (cmeister747, flores_core) | 151,343 | 46.472 | 0.027 | 0.251 | 3402.5 | 0.197 | 0.260 | 0.000 | --- | --- | --- | 13 | 0.0630 | 11.71 | flores_core | cmeister747 | 2026-02-23 |
| GPT OSS (201k) (cmeister747, flores_core) | 200,019 | 34.730 | 0.033 | 0.245 | 6659.5 | 0.065 | 0.779 | 0.354 | --- | --- | --- | 13 | 0.1087 | 13.22 | flores_core | cmeister747 | 2026-02-23 |
| aggresssive_N2_160000 (vkanjira, flores_core) | 19,051 | 62.586 | 0.019 | 0.295 | 382.8 | 0.120 | 0.635 | 0.382 | --- | --- | --- | 13 | 0.0069 | 9.53 | flores_core | vkanjira | 2026-02-22 |
| Boundless pairwise aggressive n2 (dkletz, flores_core) | 5,033 | 75.587 | 0.017 | **0.480** | **121.7** | 0.131 | --- | 0.087 | --- | --- | --- | 13 | 0.0025 | 7.88 | flores_core | dkletz | 2026-02-22 |
