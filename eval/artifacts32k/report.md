# kototok Benchmark Report (vocab_size=32768)

## 1. Compression (`chars/token`, `bytes/token`, `tokens`)

| Model | JA c/t | JA b/t | JA tokens | EN c/t | Code c/t | Mixed c/t |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| kototok:asym-full | 5.696 | 16.397 | 12202 | 21.330 | 3.897 | 3.051 |
| kototok:asym-no-phase2 | 2.818 | 8.113 | 24662 | 5.750 | 2.812 | 2.162 |
| kototok:asym-no-picky | 5.696 | 16.397 | 12202 | 21.330 | 3.897 | 3.052 |
| kototok:asym-no-acsg | 5.696 | 16.397 | 12202 | 21.330 | 3.897 | 3.051 |
| kototok:strict-script | 1.762 | 5.072 | 39446 | 5.750 | 2.812 | 2.072 |
| kototok:standard-bbpe | 5.313 | 15.294 | 13082 | 5.750 | 2.812 | 2.320 |
| hf-bpe | 5.529 | 15.918 | 12569 | 5.888 | 3.074 | 2.822 |
| hf-unigram | 2.166 | 6.235 | 32087 | 3.818 | 2.072 | 1.608 |
| tiktoken:cl100k_base (external-100k) | 0.852 | 2.453 | 81562 | 5.761 | 3.473 | 1.945 |
| tiktoken:o200k_base (external-200k) | 1.166 | 3.355 | 59627 | 5.790 | 3.508 | 2.214 |

## 2. UTF-8 Validity & Dead Token Rate

| Model | Vocab | Learned Non-Fallback | Invalid UTF-8 | Invalid Rate (%) | Dead Tokens (%) |
| --- | ---: | ---: | ---: | ---: | ---: |
| kototok:asym-full | 27528 | 27400 | 0 | 0.00% | 85.02% |
| kototok:asym-no-phase2 | 3616 | 3488 | 0 | 0.00% | 51.11% |
| kototok:asym-no-picky | 28687 | 28559 | 0 | 0.00% | 85.62% |
| kototok:asym-no-acsg | 27527 | 27399 | 0 | 0.00% | 85.01% |
| kototok:strict-script | 4204 | 4076 | 0 | 0.00% | 62.80% |
| kototok:standard-bbpe | 20307 | 20179 | 863 | 4.28% | 81.67% |
| hf-bpe | 25687 | 25431 | 902 | 3.55% | 85.42% |
| hf-unigram | 9443 | 9442 | 0 | 0.00% | 80.85% |
| tiktoken:cl100k_base (external-100k) | 100277 | 100000 | 645 | 0.65% | 98.66% |
| tiktoken:o200k_base (external-200k) | 200019 | 199742 | 1434 | 0.72% | 99.26% |

## 3. Japanese Morphological Boundary & Okurigana Evaluation

| Model | Precision | Recall | F1 | Okurigana Integrity (%) | Noun+Particle Over-Merge (%) |
| --- | ---: | ---: | ---: | ---: | ---: |
| kototok:asym-full | 0.4546 | 0.9503 | 0.6150 | 48.16% | 11.53% |
| kototok:asym-no-phase2 | 0.4553 | 0.9584 | 0.6173 | 44.90% | 9.52% |
| kototok:asym-no-picky | 0.4815 | 0.9430 | 0.6375 | 48.16% | 11.03% |
| kototok:asym-no-acsg | 0.4546 | 0.9503 | 0.6150 | 48.16% | 11.53% |
| kototok:strict-script | 0.4689 | 0.9845 | 0.6353 | 0.00% | 0.00% |
| kototok:standard-bbpe | 0.4001 | 0.8126 | 0.5362 | 38.78% | 2.76% |
| hf-bpe | 0.4061 | 0.8158 | 0.5423 | 39.18% | 2.76% |
| hf-unigram | 0.6362 | 0.9633 | 0.7663 | 47.76% | 8.52% |
| tiktoken:cl100k_base (external-100k) | 0.4262 | 0.9951 | 0.5968 | 0.00% | 0.00% |
| tiktoken:o200k_base (external-200k) | 0.5971 | 0.9845 | 0.7434 | 2.45% | 0.75% |

## 4. Throughput (1-thread) & Adversarial 1 MiB Latency

| Model | JA Throughput (MB/s) | Adversarial 1 MiB (ms) | Adversarial Throughput (MB/s) |
| --- | ---: | ---: | ---: |
| kototok:asym-full | 54.48 | 15.18 | 69.09 |
| kototok:asym-no-phase2 | 73.95 | 17.94 | 58.46 |
| kototok:asym-no-picky | 68.23 | 10.98 | 95.46 |
| kototok:asym-no-acsg | 58.67 | 14.39 | 72.87 |
| kototok:strict-script | 75.75 | 8.26 | 126.96 |
| kototok:standard-bbpe | 142.72 | 28.53 | 36.76 |
| hf-bpe | 3.09 | 385.28 | 2.72 |
| hf-unigram | 3.87 | 362.47 | 2.89 |
| tiktoken:cl100k_base (external-100k) | 16.05 | 100.59 | 10.42 |
| tiktoken:o200k_base (external-200k) | 16.57 | 86.56 | 12.11 |
