# Tokenizer Evaluation Results

_Last updated: 2026-03-02 07:37:03_

| Tokenizer | Vocab Size | Fertility ↓ | Compression Rate ↑ | Vocab Util. ↑ | Avg Token Rank ↓ | Gini ↓ | 3-Digit Align. F1 ↑ | Op. Isolation ↑ | AST Align. ↑ | Ident. Frag. ↓ | Depth Corr. ↑ | Pat. Stability ↑ | Bound. Cross ↓ | Char Split ↓ | Languages | Dataset | User | Date |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Classical (meistecl, flores_core) | 128,004 | 36.127 | 0.031 | 0.369 | 6524.2 | 0.049 | 0.779 | 0.190 | 0.148 | 0.055 | **0.000** | **1.000** | 0.0033 | 0.0052 | 13 | flores_core | meistecl | 2026-03-01 |
| Parity-aware (meistecl, flores_core) | 128,004 | 38.139 | 0.029 | 0.322 | 5367.5 | **0.012** | 0.742 | 0.191 | 0.188 | 0.082 | 0.000 | 1.000 | 0.0037 | 0.0058 | 13 | flores_core | meistecl | 2026-03-01 |
| Parity-aware (hybrid) (meistecl, flores_core) | 128,004 | 37.385 | 0.030 | 0.330 | 5693.5 | 0.028 | 0.753 | 0.190 | 0.160 | 0.066 | 0.000 | 1.000 | 0.0036 | 0.0057 | 13 | flores_core | meistecl | 2026-03-01 |
| Gemma 3 (meistecl, flores_core) | 262,145 | 34.075 | 0.033 | 0.217 | 7779.7 | 0.054 | 0.446 | 0.346 | 0.088 | 0.030 | -0.113 | 0.966 | 0.0426 | 0.0062 | 13 | flores_core | meistecl | 2026-03-01 |
| Qwen 3 (meistecl, flores_core) | 151,669 | 44.867 | 0.029 | 0.228 | 3303.4 | 0.181 | 0.379 | 0.527 | 0.605 | 0.383 | 0.000 | 1.000 | 0.0513 | 0.0759 | 13 | flores_core | meistecl | 2026-03-01 |
| Qwen 3.5 (meistecl, flores_core) | 248,077 | 33.854 | **0.035** | 0.234 | 7687.9 | 0.099 | 0.379 | 0.527 | **0.680** | 0.381 | 0.000 | 1.000 | 0.0009 | 0.0021 | 13 | flores_core | meistecl | 2026-03-01 |
| GPT-4 (meistecl, flores_core) | 100,261 | 55.839 | 0.023 | 0.192 | 1391.9 | 0.204 | 0.760 | 0.531 | 0.610 | 0.434 | 0.000 | 1.000 | 0.0510 | 0.1586 | 13 | flores_core | meistecl | 2026-03-01 |
| GPT-4o (meistecl, flores_core) | 200,000 | 36.773 | 0.031 | 0.226 | 5833.0 | 0.059 | 0.756 | 0.364 | 0.615 | 0.450 | 0.000 | 1.000 | 0.0089 | 0.0130 | 13 | flores_core | meistecl | 2026-03-01 |
| Mistral-Nemo (meistecl, flores_core) | 131,072 | 37.419 | 0.031 | 0.341 | 5610.9 | 0.069 | 0.379 | 0.139 | 0.060 | 0.037 | 0.000 | 1.000 | 0.0014 | 0.0119 | 13 | flores_core | meistecl | 2026-03-01 |
| DeepSeek R1 (meistecl, flores_core) | 128,815 | 41.449 | 0.029 | 0.285 | 3944.5 | 0.136 | 0.779 | 0.188 | 0.000 | --- | 0.000 | 1.000 | 0.0071 | 0.0126 | 13 | flores_core | meistecl | 2026-03-01 |
| Cohere Command R (meistecl, flores_core) | 255,029 | 36.519 | 0.033 | 0.233 | 7108.9 | 0.144 | 0.379 | 0.361 | 0.032 | 0.016 | 0.000 | 1.000 | 0.0005 | 0.0004 | 13 | flores_core | meistecl | 2026-03-01 |
| Kimi (meistecl, flores_core) | 163,840 | 45.112 | 0.027 | 0.173 | 2588.9 | 0.137 | 0.779 | 0.489 | 0.609 | 0.408 | 0.000 | 1.000 | 0.0116 | 0.0252 | 13 | flores_core | meistecl | 2026-03-01 |
| GLM (meistecl, flores_core) | 151,343 | 46.472 | 0.027 | 0.251 | 3402.5 | 0.197 | 0.270 | 0.000 | 0.001 | **0.000** | 0.000 | 1.000 | **0.0000** | **0.0000** | 13 | flores_core | meistecl | 2026-03-01 |
| GPT OSS (meistecl, flores_core) | 200,019 | 34.730 | 0.033 | 0.245 | 6659.5 | 0.065 | 0.779 | 0.354 | 0.609 | 0.401 | 0.000 | 1.000 | 0.0019 | 0.0059 | 13 | flores_core | meistecl | 2026-03-01 |
| Apertus (meistecl, flores_core) | 131,072 | 37.419 | 0.031 | 0.341 | 5610.9 | 0.069 | 0.379 | 0.139 | 0.060 | 0.037 | 0.000 | 1.000 | 0.0014 | 0.0119 | 13 | flores_core | meistecl | 2026-03-01 |
| SuperBPE default regex (meistecl, flores_core) | 130,000 | 40.307 | 0.028 | 0.241 | 4421.4 | 0.069 | 0.779 | 0.379 | 0.225 | 0.028 | 0.000 | 1.000 | 0.0050 | 0.0205 | 13 | flores_core | meistecl | 2026-03-01 |
| DeepSeek V3 (meistecl, flores_core) | 128,815 | 41.449 | 0.029 | 0.285 | 3944.5 | 0.136 | 0.779 | 0.188 | 0.000 | --- | 0.000 | 1.000 | 0.0071 | 0.0126 | 13 | flores_core | meistecl | 2026-03-01 |
| SuperBPE Apertus regex (meistecl, flores_core) | 130,000 | 42.557 | 0.028 | 0.253 | 4104.3 | 0.122 | 0.779 | 0.966 | 0.259 | 0.029 | 0.000 | 1.000 | 0.0038 | 0.0156 | 13 | flores_core | meistecl | 2026-03-01 |
| SuperBPE qwen regex (meistecl, flores_core) | 130,000 | 42.575 | 0.027 | 0.280 | 5048.7 | 0.059 | 0.379 | 0.312 | 0.284 | 0.024 | 0.000 | 1.000 | 0.0047 | 0.0196 | 13 | flores_core | meistecl | 2026-03-01 |
| SuperBPE math (meistecl, flores_core) | 130,000 | 43.112 | 0.028 | 0.254 | 3831.6 | 0.131 | 0.779 | **0.967** | 0.264 | 0.028 | 0.000 | 1.000 | 0.0047 | 0.0193 | 13 | flores_core | meistecl | 2026-03-01 |
| SuperBPE math new reg (meistecl, flores_core) | 130,000 | 40.223 | 0.029 | 0.258 | 4403.1 | 0.085 | **0.953** | 0.311 | 0.230 | 0.027 | 0.000 | 1.000 | 0.0061 | 0.0247 | 13 | flores_core | meistecl | 2026-03-01 |
| PA BPE grouped (meistecl, flores_core) | 120,260 | 45.497 | 0.025 | 0.199 | 2898.5 | 0.073 | 0.712 | 0.191 | 0.159 | 0.067 | 0.000 | 1.000 | 0.0105 | 0.0415 | 13 | flores_core | meistecl | 2026-03-01 |
| Apertus_Coarse_Family (ayavuz, flores_core) | --- | 40.070 | 0.029 | 0.340 | --- | 0.103 | --- | --- | --- | --- | --- | --- | --- | --- | 13 | flores_core | ayavuz | 2026-02-21 |
| Apertus_Fine_Family (ayavuz, flores_core) | --- | 40.619 | 0.029 | 0.336 | --- | 0.114 | --- | --- | --- | --- | --- | --- | --- | --- | 13 | flores_core | ayavuz | 2026-02-21 |
| Apertus_Per_Lang (ayavuz, flores_core) | --- | 38.592 | 0.030 | 0.333 | --- | 0.083 | --- | --- | --- | --- | --- | --- | --- | --- | 13 | flores_core | ayavuz | 2026-02-21 |
| BPE_Coarse_Family (ayavuz, flores_core) | --- | 42.935 | 0.027 | 0.245 | --- | 0.083 | --- | --- | --- | --- | --- | --- | --- | --- | 13 | flores_core | ayavuz | 2026-02-21 |
| BPE_Fine_Family (ayavuz, flores_core) | --- | 44.451 | 0.026 | 0.237 | --- | 0.082 | --- | --- | --- | --- | --- | --- | --- | --- | 13 | flores_core | ayavuz | 2026-02-21 |
| BPE_Per_Lang (ayavuz, flores_core) | --- | 42.585 | 0.027 | 0.228 | --- | 0.082 | --- | --- | --- | --- | --- | --- | --- | --- | 13 | flores_core | ayavuz | 2026-02-21 |
| Unigram_Coarse_Family (ayavuz, flores_core) | --- | 47.868 | 0.024 | 0.254 | --- | 0.066 | --- | --- | --- | --- | --- | --- | --- | --- | 13 | flores_core | ayavuz | 2026-02-21 |
| Unigram_Fine_Family (ayavuz, flores_core) | --- | 47.843 | 0.024 | 0.247 | --- | 0.065 | --- | --- | --- | --- | --- | --- | --- | --- | 13 | flores_core | ayavuz | 2026-02-21 |
| Unigram_Per_Lang (ayavuz, flores_core) | --- | 47.415 | 0.024 | 0.233 | --- | 0.061 | --- | --- | --- | --- | --- | --- | --- | --- | 13 | flores_core | ayavuz | 2026-02-21 |
| Zip2Zip/Gemma 3 (512 codebook) (saibo, flores_core) | 262,657 | **33.668** | 0.034 | 0.217 | 7808.0 | 0.053 | --- | --- | --- | --- | --- | --- | --- | --- | 13 | flores_core | saibo | 2026-02-18 |
| Zip2Zip/Qwen 3 (512 codebook) (saibo, flores_core) | 152,181 | 43.280 | 0.029 | 0.228 | 3441.9 | 0.164 | --- | --- | --- | --- | --- | --- | --- | --- | 13 | flores_core | saibo | 2026-02-18 |
| Zip2Zip/Mistral-Nemo (512 codebook) (saibo, flores_core) | 131,584 | 36.910 | 0.031 | 0.340 | 5639.2 | 0.068 | --- | --- | --- | --- | --- | --- | --- | --- | 13 | flores_core | saibo | 2026-02-18 |
| Zip2Zip/DeepSeek R1 (512 codebook) (saibo, flores_core) | 129,327 | 40.814 | 0.029 | 0.285 | 3985.5 | 0.133 | --- | --- | --- | --- | --- | --- | --- | --- | 13 | flores_core | saibo | 2026-02-18 |
| Zip2Zip/GPT OSS (512 codebook) (saibo, flores_core) | 200,531 | 34.314 | 0.033 | 0.244 | 6676.6 | 0.064 | --- | --- | --- | --- | --- | --- | --- | --- | 13 | flores_core | saibo | 2026-02-18 |
| Zip2Zip/Apertus (512 codebook) (saibo, flores_core) | 131,584 | 36.910 | 0.031 | 0.340 | 5639.2 | 0.068 | --- | --- | --- | --- | --- | --- | --- | --- | 13 | flores_core | saibo | 2026-02-18 |
| aggresssive_N2_160000 (vkanjira, flores_core) | 19,051 | 62.586 | 0.019 | 0.295 | 382.8 | 0.120 | 0.635 | 0.382 | --- | --- | --- | --- | --- | --- | 13 | flores_core | vkanjira | 2026-02-22 |
| Boundless pairwise aggressive n2 (dkletz, flores_core) | 5,033 | 75.587 | 0.017 | **0.480** | **121.7** | 0.131 | --- | 0.087 | --- | --- | --- | --- | --- | --- | 13 | flores_core | dkletz | 2026-02-22 |
