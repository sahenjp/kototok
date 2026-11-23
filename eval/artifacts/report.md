# kototok Benchmark Report (vocab_size=4096)

## 1. Compression (`chars/token`, `bytes/token`, `tokens`)

| Model | JA c/t | JA b/t | JA tokens | EN c/t | Code c/t | Mixed c/t |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| kototok:asym-full | 4.608 | 13.304 | 15040 | 18.795 | 3.863 | 2.909 |
| kototok:asym-no-phase2 | 2.751 | 7.942 | 25192 | 5.755 | 2.811 | 2.158 |
| kototok:asym-no-picky | 4.078 | 11.773 | 16996 | 15.695 | 3.699 | 2.857 |
| kototok:asym-no-acsg | 4.608 | 13.304 | 15040 | 18.795 | 3.863 | 2.909 |
| kototok:strict-script | 1.710 | 4.936 | 40535 | 5.755 | 2.811 | 2.068 |
| kototok:standard-bbpe | 4.261 | 12.301 | 16266 | 5.755 | 2.782 | 2.249 |
| hf-bpe | 4.343 | 12.538 | 15958 | 5.899 | 3.038 | 2.694 |
| hf-unigram | 2.070 | 5.975 | 33489 | 3.835 | 2.085 | 1.610 |
| tiktoken:cl100k_base (external-100k) | 0.856 | 2.471 | 80976 | 5.771 | 3.500 | 1.957 |
| tiktoken:o200k_base (external-200k) | 1.155 | 3.334 | 60007 | 5.803 | 3.534 | 2.228 |

## 2. UTF-8 Validity & Dead Token Rate

| Model | Vocab | Learned Non-Fallback | Invalid UTF-8 | Invalid Rate (%) | Dead Tokens (%) |
| --- | ---: | ---: | ---: | ---: | ---: |
| kototok:asym-full | 4096 | 3968 | 0 | 0.00% | 56.35% |
| kototok:asym-no-phase2 | 2899 | 2771 | 0 | 0.00% | 47.43% |
| kototok:asym-no-picky | 4096 | 3968 | 0 | 0.00% | 67.90% |
| kototok:asym-no-acsg | 4096 | 3968 | 0 | 0.00% | 56.35% |
| kototok:strict-script | 3474 | 3346 | 0 | 0.00% | 61.80% |
| kototok:standard-bbpe | 4096 | 3968 | 864 | 21.77% | 58.57% |
| hf-bpe | 4096 | 3840 | 913 | 23.78% | 57.18% |
| hf-unigram | 3867 | 3866 | 0 | 0.00% | 61.08% |
| tiktoken:cl100k_base (external-100k) | 100277 | 100000 | 645 | 0.65% | 98.71% |
| tiktoken:o200k_base (external-200k) | 200019 | 199742 | 1434 | 0.72% | 99.28% |

## 3. Japanese Morphological Boundary & Okurigana Evaluation

| Model | Precision | Recall | F1 | Okurigana Integrity (%) | Noun+Particle Over-Merge (%) |
| --- | ---: | ---: | ---: | ---: | ---: |
| kototok:asym-full | 0.4556 | 0.9544 | 0.6168 | 48.16% | 10.28% |
| kototok:asym-no-phase2 | 0.4553 | 0.9584 | 0.6173 | 44.90% | 9.52% |
| kototok:asym-no-picky | 0.4824 | 0.9519 | 0.6404 | 47.35% | 8.02% |
| kototok:asym-no-acsg | 0.4556 | 0.9544 | 0.6168 | 48.16% | 10.28% |
| kototok:strict-script | 0.4689 | 0.9845 | 0.6353 | 0.00% | 0.00% |
| kototok:standard-bbpe | 0.3924 | 0.8134 | 0.5294 | 20.00% | 3.01% |
| hf-bpe | 0.3966 | 0.8158 | 0.5337 | 20.00% | 3.01% |
| hf-unigram | 0.6341 | 0.9633 | 0.7648 | 46.94% | 8.52% |
| tiktoken:cl100k_base (external-100k) | 0.4262 | 0.9951 | 0.5968 | 0.00% | 0.00% |
| tiktoken:o200k_base (external-200k) | 0.5971 | 0.9845 | 0.7434 | 2.45% | 0.75% |

## 4. Throughput (1-thread) & Adversarial 1 MiB Latency

| Model | JA Throughput (MB/s) | Adversarial 1 MiB (ms) | Adversarial Throughput (MB/s) |
| --- | ---: | ---: | ---: |
| kototok:asym-full | 83.26 | 11.43 | 91.76 |
| kototok:asym-no-phase2 | 92.21 | 12.97 | 80.87 |
| kototok:asym-no-picky | 90.78 | 11.86 | 88.39 |
| kototok:asym-no-acsg | 84.91 | 14.52 | 72.20 |
| kototok:strict-script | 65.88 | 9.43 | 111.21 |
| kototok:standard-bbpe | 144.97 | 30.10 | 34.83 |
| hf-bpe | 1.75 | 552.06 | 1.90 |
| hf-unigram | 3.20 | 368.78 | 2.84 |
| tiktoken:cl100k_base (external-100k) | 14.68 | 102.57 | 10.22 |
| tiktoken:o200k_base (external-200k) | 15.83 | 96.10 | 10.91 |
