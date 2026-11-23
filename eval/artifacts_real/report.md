# kototok Benchmark Report (vocab_size=32768)

## 1. Compression (`chars/token`, `bytes/token`, `tokens`)

| Model | JA c/t | JA b/t | JA tokens | EN c/t | Code c/t | Mixed c/t |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| kototok:asym-full | 1.754 | 4.807 | 77629 | 4.340 | 4.246 | 3.534 |
| kototok:asym-no-phase2 | 1.719 | 4.713 | 79187 | 3.389 | 3.242 | 2.943 |
| kototok:asym-no-picky | 1.765 | 4.838 | 77146 | 4.274 | 4.289 | 3.562 |
| kototok:asym-no-acsg | 1.775 | 4.865 | 76705 | 4.213 | 4.264 | 3.553 |
| kototok:strict-script | 1.639 | 4.492 | 83069 | 3.387 | 3.242 | 2.924 |
| kototok:standard-bbpe | 1.746 | 4.788 | 77945 | 3.375 | 3.225 | 2.937 |
| hf-bpe | 1.784 | 4.891 | 76309 | 3.490 | 3.520 | 3.164 |
| hf-unigram | 1.666 | 4.566 | 81729 | 3.823 | 3.199 | 2.850 |
| tiktoken:cl100k_base (external-100k) | 0.909 | 2.491 | 149813 | 3.945 | 4.371 | 3.276 |
| tiktoken:o200k_base (external-200k) | 1.253 | 3.434 | 108676 | 3.957 | 4.350 | 3.494 |

## 2. UTF-8 Validity & Dead Token Rate

| Model | Vocab | Learned Non-Fallback | Invalid UTF-8 | Invalid Rate (%) | Dead Tokens (%) |
| --- | ---: | ---: | ---: | ---: | ---: |
| kototok:asym-full | 32768 | 32640 | 0 | 0.00% | 52.85% |
| kototok:asym-no-phase2 | 32768 | 32640 | 0 | 0.00% | 56.86% |
| kototok:asym-no-picky | 32768 | 32640 | 0 | 0.00% | 52.05% |
| kototok:asym-no-acsg | 32768 | 32640 | 0 | 0.00% | 50.24% |
| kototok:strict-script | 32768 | 32640 | 0 | 0.00% | 58.50% |
| kototok:standard-bbpe | 32768 | 32640 | 701 | 2.15% | 53.36% |
| hf-bpe | 32768 | 32512 | 705 | 2.17% | 52.69% |
| hf-unigram | 32768 | 32767 | 0 | 0.00% | 54.07% |
| tiktoken:cl100k_base (external-100k) | 100277 | 100000 | 645 | 0.65% | 89.11% |
| tiktoken:o200k_base (external-200k) | 200019 | 199742 | 1434 | 0.72% | 93.64% |

## 3. Japanese Morphological Boundary & Okurigana Evaluation

| Model | Precision | Recall | F1 | Okurigana Integrity (%) | Noun+Particle Over-Merge (%) |
| --- | ---: | ---: | ---: | ---: | ---: |
| kototok:asym-full | 0.8306 | 0.9593 | 0.8903 | 55.10% | 0.25% |
| kototok:asym-no-phase2 | 0.7859 | 0.9332 | 0.8532 | 36.33% | 7.02% |
| kototok:asym-no-picky | 0.7624 | 0.9022 | 0.8264 | 31.43% | 8.77% |
| kototok:asym-no-acsg | 0.7528 | 0.8264 | 0.7879 | 46.53% | 38.35% |
| kototok:strict-script | 0.7353 | 0.9576 | 0.8319 | 0.00% | 0.00% |
| kototok:standard-bbpe | 0.7059 | 0.7726 | 0.7377 | 38.37% | 19.55% |
| hf-bpe | 0.7080 | 0.7726 | 0.7389 | 38.37% | 19.55% |
| hf-unigram | 0.7753 | 0.8859 | 0.8269 | 43.67% | 11.28% |
| tiktoken:cl100k_base (external-100k) | 0.4262 | 0.9951 | 0.5968 | 0.00% | 0.00% |
| tiktoken:o200k_base (external-200k) | 0.5971 | 0.9845 | 0.7434 | 2.45% | 0.75% |

## 4. Throughput (1-thread) & Adversarial 1 MiB Latency

| Model | JA Throughput (MB/s) | Adversarial 1 MiB (ms) | Adversarial Throughput (MB/s) |
| --- | ---: | ---: | ---: |
| kototok:asym-full | 73.63 | 9.04 | 116.00 |
| kototok:asym-no-phase2 | 78.48 | 9.10 | 115.19 |
| kototok:asym-no-picky | 76.03 | 8.96 | 117.03 |
| kototok:asym-no-acsg | 76.86 | 8.73 | 120.08 |
| kototok:strict-script | 90.92 | 6.05 | 173.28 |
| kototok:standard-bbpe | 116.78 | 31.02 | 33.80 |
| hf-bpe | 3.31 | 391.49 | 2.68 |
| hf-unigram | 5.98 | 273.37 | 3.84 |
| tiktoken:cl100k_base (external-100k) | 16.46 | 78.77 | 13.31 |
| tiktoken:o200k_base (external-200k) | 18.09 | 75.84 | 13.83 |
