# TransVoice CPU benchmark

> All numbers are CPU-only (no GPU). Accuracy runs were executed in parallel, so their speed columns are pessimistic unless marked `*`. The **office-PC profile** (`CPUs = 12,13,14,15`: four E-cores of an i5-14600K, machine otherwise idle) is the reference for deployment speed.

## ASR (CPU, 2 threads)

Error = CER for Japanese, WER for English/Vietnamese (normalized, lower is better). RTF = processing time / audio time. Chunk p95 = compute per 560 ms chunk (streaming only). `*` = timing measured on an idle machine.

### fleurs / ja (CER, n=150)

| model | CER % | RTF | p95 ms/utt | chunk p95 ms | RAM MB | CPU cores |
|---|---|---|---|---|---|---|
| parakeet-ja* | 5.39 | 0.1801 | 3674 |  | 793 | 1.89 |
| sensevoice* | 7.43 | 0.0939 | 1905 |  | 354 | 1.89 |
| reazonspeech-k2* | 7.81 | 0.0412 | 843 |  | 456 | 3.14 |
| qwen3-asr | 13.3 | 0.4758 | 12105 |  | 2591 | 2.1 |
| nemotron | 14.74 | 0.3197 | 6369 | 207.9 | 746 | 3.72 |

### reazon / ja (CER, n=300)

| model | CER % | RTF | p95 ms/utt | chunk p95 ms | RAM MB | CPU cores |
|---|---|---|---|---|---|---|
| reazonspeech-k2 | 5.31 | 0.0333 | 464 |  | 626 | 3.73 |
| parakeet-ja | 9.98 | 0.0984 | 1265 |  | 887 | 1.98 |
| sensevoice | 25.09 | 0.0371 | 485 |  | 417 | 1.98 |
| nemotron | 29.62 | 0.4542 | 6077 | 352.8 | 749 | 3.69 |

### fleurs / en (WER, n=150)

| model | WER % | RTF | p95 ms/utt | chunk p95 ms | RAM MB | CPU cores |
|---|---|---|---|---|---|---|
| sensevoice* | 8.65 | 0.0926 | 1459 |  | 370 | 1.9 |
| nemotron | 14.02 | 0.3577 | 6797 | 226.6 | 745 | 3.86 |

### fleurs / vi (WER, n=150)

| model | WER % | RTF | p95 ms/utt | chunk p95 ms | RAM MB | CPU cores |
|---|---|---|---|---|---|---|
| zipformer-vi-30m* | 9.61 | 0.0244 | 554 |  | 329 | 3.63 |
| vietasr* | 11.24 | 0.0301 | 668 |  | 377 | 3.46 |
| nemotron | 15.94 | 0.4164 | 8402 | 281.5 | 742 | 3.77 |

## Spoken language ID (JA/EN/VI, silence trimmed)

| detector | audio heard | accuracy % | ja % | en % | vi % | p50 ms |
|---|---|---|---|---|---|---|
| whisper-tiny | 1 | 45.7 | 25.0 | 88.0 | 24.0 | 91 |
| whisper-tiny | 2 | 71.0 | 48.0 | 97.0 | 68.0 | 130 |
| whisper-tiny | 3 | 89.7 | 85.0 | 99.0 | 85.0 | 175 |
| whisper-tiny | full | 99.7 | 100.0 | 100.0 | 99.0 | 343 |
| whisper-base | 1 | 50.0 | 27.0 | 92.0 | 31.0 | 221 |
| whisper-base | 2 | 71.0 | 48.0 | 90.0 | 75.0 | 259 |
| whisper-base | 3 | 93.7 | 88.0 | 98.0 | 95.0 | 311 |
| whisper-base | full | 100.0 | 100.0 | 100.0 | 100.0 | 628 |
| sensevoice | 1 | 49.0 | 29.0 | 100.0 | 18.0 | 87 |
| sensevoice | 2 | 64.7 | 60.0 | 100.0 | 34.0 | 133 |
| sensevoice | 3 | 83.3 | 93.0 | 100.0 | 57.0 | 192 |
| sensevoice | full | 89.3 | 100.0 | 100.0 | 68.0 | 640 |

## Translation (CPU, 4 threads)

COMET = wmt22-comet-da (higher is better). Bad % = COMET < 0.65 or a broken-output flag (empty, wrong language, too short/long, repetition, commentary). `*` = timing measured on an idle machine.

### en → ja

| engine | COMET | bad % | chrF | flag % | p50 ms | p95 ms | RAM MB | CPU cores |
|---|---|---|---|---|---|---|---|---|
| hy-mt-1.8b-iq4nl | 0.9197 | 0.0 | 38.19 | 0.0 | 1464 | 2398 | 2042 | 3.97 |
| hy-mt-1.8b-q4_0 | 0.9195 | 0.0 | 38.22 | 0.0 | 1380 | 2514 | 2032 | 3.97 |
| translategemma-4b | 0.9182 | 0.0 | 40.02 | 0.0 | 7568 | 12266 | 7729 | 3.9 |
| hy-mt-1.8b | 0.918 | 0.0 | 38.18 | 0.0 | 3758 | 6505 | 3805 | 3.91 |
| shisa-1.2b | 0.913 | 1.0 | 39.95 | 1.0 | 6476 | 13040 | 1263 | 3.96 |
| qwen3.5-4b | 0.9112 | 0.0 | 36.43 | 0.0 | 6418 | 10414 | 4167 | 3.98 |
| lfm2-350m-enjp | 0.9098 | 0.0 | 41.86 | 0.0 | 681 | 1307 | 432 | 3.93 |
| cat-translate-1.4b | 0.9011 | 3.0 | 35.91 | 2.0 | 2368 | 3889 | 1433 | 3.85 |
| qwen3.5-2b | 0.8938 | 1.0 | 31.25 | 1.0 | 3463 | 6019 | 1881 | 3.95 |
| nllb-1.3b | 0.8879 | 6.0 | 34.16 | 4.0 | 2122 | 5450 | 1575 | 3.83 |
| nllb-600m | 0.8769 | 3.0 | 30.58 | 2.0 | 1126 | 2288 | 756 | 3.94 |

### en → vi

| engine | COMET | bad % | chrF | flag % | p50 ms | p95 ms | RAM MB | CPU cores |
|---|---|---|---|---|---|---|---|---|
| qwen3.5-4b | 0.8962 | 0.0 | 57.7 | 0.0 | 4695 | 7999 | 4167 | 3.99 |
| translategemma-4b | 0.896 | 1.0 | 56.06 | 1.0 | 8453 | 14798 | 7677 | 3.91 |
| hy-mt-1.8b | 0.8889 | 2.0 | 52.91 | 2.0 | 4159 | 7070 | 4407 | 3.93 |
| hy-mt-1.8b-q4_0 | 0.8887 | 0.0 | 53.27 | 0.0 | 1654 | 2740 | 2032 | 3.97 |
| hy-mt-1.8b-iq4nl | 0.888 | 4.0 | 53.03 | 4.0 | 1630 | 2633 | 2042 | 3.97 |
| nllb-1.3b | 0.8815 | 4.0 | 59.81 | 2.0 | 2404 | 4533 | 1552 | 3.87 |
| qwen3.5-2b | 0.8783 | 1.0 | 52.9 | 0.0 | 3827 | 6408 | 1881 | 3.96 |
| nllb-600m | 0.8667 | 5.0 | 56.4 | 1.0 | 1268 | 2421 | 757 | 3.96 |

### ja → en

| engine | COMET | bad % | chrF | flag % | p50 ms | p95 ms | RAM MB | CPU cores |
|---|---|---|---|---|---|---|---|---|
| qwen3.5-4b | 0.8846 | 0.0 | 58.62 | 0.0 | 8938 | 16026 | 4167 | 3.81 |
| translategemma-4b | 0.8794 | 0.0 | 56.54 | 0.0 | 5807 | 9733 | 5397 | 3.96 |
| nllb-1.3b | 0.8777 | 0.0 | 59.43 | 0.0 | 1888 | 3684 | 1561 | 3.9 |
| lfm2-350m-enjp | 0.8739 | 0.0 | 58.47 | 0.0 | 645 | 1179 | 432 | 3.87 |
| hy-mt-1.8b-iq4nl | 0.8736 | 0.0 | 56.2 | 0.0 | 1125 | 1880 | 2042 | 3.97 |
| hy-mt-1.8b | 0.8723 | 0.0 | 55.68 | 0.0 | 2368 | 4019 | 2539 | 3.95 |
| shisa-1.2b | 0.8717 | 1.0 | 56.06 | 0.0 | 3588 | 7479 | 1263 | 3.96 |
| hy-mt-1.8b-q4_0 | 0.8704 | 1.0 | 55.32 | 1.0 | 1111 | 1899 | 2031 | 3.97 |
| qwen3.5-2b | 0.8683 | 0.0 | 55.17 | 0.0 | 4762 | 8034 | 1881 | 3.65 |
| cat-translate-1.4b | 0.8643 | 2.0 | 55.67 | 2.0 | 3205 | 5580 | 1422 | 3.61 |
| nllb-600m | 0.8569 | 3.0 | 54.72 | 2.0 | 1146 | 2153 | 744 | 3.89 |

### ja → vi

| engine | COMET | bad % | chrF | flag % | p50 ms | p95 ms | RAM MB | CPU cores |
|---|---|---|---|---|---|---|---|---|
| qwen3.5-4b | 0.8791 | 0.0 | 48.75 | 0.0 | 9128 | 15338 | 4167 | 3.97 |
| translategemma-4b | 0.8782 | 2.0 | 47.25 | 2.0 | 8259 | 14332 | 7071 | 3.94 |
| hy-mt-1.8b-iq4nl | 0.8693 | 1.0 | 45.03 | 1.0 | 1800 | 3559 | 2042 | 3.97 |
| hy-mt-1.8b-iq4nl-pivot | 0.8679 | 1.0 | 45.81 | 1.0 | 4526 | 8316 | 2042 | 3.96 |
| hy-mt-1.8b-q4_0 | 0.8663 | 2.0 | 44.86 | 2.0 | 1752 | 2952 | 2032 | 3.98 |
| hy-mt-1.8b | 0.865 | 1.0 | 44.61 | 1.0 | 3830 | 7568 | 3255 | 3.96 |
| nllb-1.3b | 0.8605 | 1.0 | 48.22 | 1.0 | 2593 | 6205 | 1565 | 3.41 |
| qwen3.5-2b | 0.8446 | 3.0 | 44.13 | 3.0 | 4390 | 8109 | 1881 | 3.94 |
| nllb-600m | 0.8426 | 2.0 | 43.82 | 0.0 | 1283 | 2237 | 751 | 3.96 |

### vi → en

| engine | COMET | bad % | chrF | flag % | p50 ms | p95 ms | RAM MB | CPU cores |
|---|---|---|---|---|---|---|---|---|
| qwen3.5-4b | 0.8796 | 0.0 | 62.48 | 0.0 | 3708 | 5982 | 4167 | 3.99 |
| nllb-1.3b | 0.8761 | 1.0 | 65.68 | 0.0 | 1939 | 3385 | 1550 | 3.89 |
| translategemma-4b | 0.8711 | 1.0 | 58.97 | 0.0 | 8190 | 15979 | 7162 | 3.82 |
| nllb-600m | 0.8688 | 1.0 | 63.39 | 0.0 | 1478 | 2790 | 752 | 3.58 |
| qwen3.5-2b | 0.8663 | 1.0 | 58.73 | 0.0 | 2786 | 4783 | 1881 | 3.97 |
| hy-mt-1.8b-iq4nl | 0.8641 | 0.0 | 57.98 | 0.0 | 1126 | 1890 | 2044 | 3.97 |
| hy-mt-1.8b | 0.8636 | 1.0 | 58.56 | 0.0 | 2798 | 5179 | 5247 | 3.87 |
| hy-mt-1.8b-q4_0 | 0.8611 | 0.0 | 57.73 | 0.0 | 1147 | 1858 | 2034 | 3.97 |

### vi → ja

| engine | COMET | bad % | chrF | flag % | p50 ms | p95 ms | RAM MB | CPU cores |
|---|---|---|---|---|---|---|---|---|
| hy-mt-1.8b-iq4nl-pivot | 0.8954 | 4.0 | 32.39 | 4.0 | 4458 | 7685 | 2044 | 3.96 |
| hy-mt-1.8b | 0.8901 | 1.0 | 31.86 | 1.0 | 4136 | 7834 | 5108 | 3.86 |
| qwen3.5-4b | 0.8895 | 0.0 | 30.93 | 0.0 | 4336 | 6724 | 4167 | 3.98 |
| translategemma-4b | 0.8893 | 1.0 | 32.31 | 1.0 | 7754 | 12982 | 7353 | 3.96 |
| hy-mt-1.8b-iq4nl | 0.8888 | 1.0 | 31.99 | 1.0 | 1596 | 2807 | 2044 | 3.97 |
| hy-mt-1.8b-q4_0 | 0.8887 | 2.0 | 31.97 | 2.0 | 1612 | 2521 | 2034 | 3.98 |
| nllb-1.3b | 0.8612 | 3.0 | 28.09 | 1.0 | 1726 | 3076 | 1550 | 3.96 |
| qwen3.5-2b | 0.8602 | 3.0 | 25.26 | 2.0 | 3237 | 5841 | 1881 | 3.97 |
| nllb-600m | 0.8511 | 4.0 | 26.44 | 0.0 | 990 | 1851 | 747 | 3.96 |

## End-to-end: speech → ASR → translation (all 6 directions)

| config | CPUs | ASR error % | COMET avg | bad % avg | e2e p50 ms | e2e p95 ms | RAM MB | CPU cores |
|---|---|---|---|---|---|---|---|---|
| A-office | 12,13,14,15 | ja:5.55 en:7.67 vi:10.77 | 0.8672 | 2.9 | 5309 | 9056 | 3460 | 3.53 |
| C-office | 12,13,14,15 | ja:5.55 en:7.67 vi:10.77 | 0.8729 | 2.1 | 10871 | 16244 | 5633 | 3.67 |

## Live app, end to end (FLEURS speech played in real time through the engine)

Final subtitles only. Lag = time from the speaker's last chunk to the final translation.

| run | CPUs | rescore | mt_fast | ASR error % | COMET | bad % | lag p50 ms | lag p90 ms |
|---|---|---|---|---|---|---|---|---|
| base | 12,13,14,15 | True | False | ja:7.84 en:16.43 | en→vi:0.8056 ja→vi:0.8622 | en:6.7 ja:6.7 | 4879 | 14695 |
| mt-fast | 12,13,14,15 | True | True | ja:7.84 en:16.43 | en→vi:0.8125 ja→vi:0.8614 | en:6.7 ja:6.7 | 4250 | 12488 |
| no-rescore | 12,13,14,15 | False | False | ja:9.86 en:17.28 | en→vi:0.8043 ja→vi:0.8426 | en:6.7 ja:0.0 | 4404 | 8085 |
| v10-max8 | 12,13,14,15 | True | True | ja:8.85 en:17.28 | en→vi:0.8734 ja→vi:0.8572 | en:0.0 ja:0.0 | 4676 | 6836 |
| v11-quiet192 | 12,13,14,15 | True | True | ja:9.61 en:14.45 | en→vi:0.8724 ja→vi:0.8546 | en:0.0 ja:0.0 | 5058 | 7061 |
| v12-period-endpause | 12,13,14,15 | True | True | ja:9.36 en:17.28 | en→vi:0.8784 ja→vi:0.8619 | en:0.0 ja:0.0 | 4800 | 6790 |
| v13-all | 12,13,14,15 | True | True | ja:9.61 en:16.15 | en→vi:0.8795 ja→vi:0.8544 | en:0.0 ja:0.0 | 4923 | 7096 |
| v14a-ramp | 12,13,14,15 | True | True | ja:8.22 en:17.0 | en→vi:0.8786 ja→vi:0.862 | en:0.0 ja:0.0 | 4698 | 7035 |
| v14b-fixed-rescore14 | 12,13,14,15 | True | True | ja:9.1 en:16.15 | en→vi:0.8795 ja→vi:0.8555 | en:0.0 ja:0.0 | 4923 | 7348 |
| v15-ramp-trust | 12,13,14,15 | True | True | ja:8.22 en:17.0 | en→vi:0.8786 ja→vi:0.8633 | en:0.0 ja:0.0 | 4754 | 7254 |
| v16-rescore16 | 12,13,14,15 | True | True | ja:6.57 en:17.0 | en→vi:0.8786 ja→vi:0.8645 | en:0.0 ja:0.0 | 4900 | 8911 |
| v17-load-guard | 12,13,14,15 | True | True | ja:6.57 en:17.0 | en→vi:0.8786 ja→vi:0.8645 | en:0.0 ja:0.0 | 4906 | 8824 |
| v2 | 12,13,14,15 | True | False | ja:9.36 en:14.73 | en→vi:0.8776 ja→vi:0.8583 | en:0.0 ja:0.0 | 8474 | 15031 |
| v3-stages | 12,13,14,15 | True | True | ja:9.36 en:14.73 | en→vi:0.8799 ja→vi:0.8574 | en:0.0 ja:0.0 | 6104 | 9310 |
| v3 | 12,13,14,15 | True | True | ja:9.36 en:14.73 | en→vi:0.8799 ja→vi:0.8574 | en:0.0 ja:0.0 | 6164 | 9140 |
| v4-early-rescore | 12,13,14,15 | True | True | ja:9.36 en:14.73 | en→vi:0.8799 ja→vi:0.8578 | en:0.0 ja:0.0 | 6868 | 10791 |
| v5-streaming-rerun | 12,13,14,15 | True | True | ja:9.36 en:14.73 | en→vi:0.8799 ja→vi:0.8574 | en:0.0 ja:0.0 | 6153 | 9500 |
| v5-streaming | 12,13,14,15 | True | True | ja:9.36 en:14.73 | en→vi:0.8801 ja→vi:0.8589 | en:0.0 ja:0.0 | 10186 | 20047 |
| v6-asr4 | 12,13,14,15 | True | True | ja:12.52 en:14.73 | en→vi:0.8762 ja→vi:0.8596 | en:0.0 ja:0.0 | 5386 | 7582 |
| v6-iq4nl | 12,13,14,15 | True | True | ja:9.36 en:14.73 | en→vi:0.8762 ja→vi:0.8584 | en:0.0 ja:0.0 | 5351 | 6879 |
| v7-gap05 | 12,13,14,15 | True | True | ja:9.36 en:15.3 | en→vi:0.8638 ja→vi:0.8584 | en:0.0 ja:0.0 | 5358 | 7505 |
| v7-norescore | 12,13,14,15 | False | True | ja:9.86 en:17.28 | en→vi:0.8784 ja→vi:0.8334 | en:0.0 ja:6.7 | 4199 | 6425 |
| v8-rescore-ja | 12,13,14,15 | True | True | ja:9.36 en:17.28 | en→vi:0.8784 ja→vi:0.8584 | en:0.0 ja:0.0 | 4428 | 6919 |
| v9-preempt | 12,13,14,15 | True | True | ja:9.36 en:17.28 | en→vi:0.8784 ja→vi:0.8584 | en:0.0 ja:0.0 | 4558 | 7041 |

## Stability

| config | minutes | utterances | errors | RAM start MB | RAM max MB | RAM growth 2nd half MB | e2e p99 ms | ASR text on non-speech | MT edge-case problems | MT critical errors |
|---|---|---|---|---|---|---|---|---|---|---|
| A-gloss-office | 3.0 | 19 | 0 | 3488 | 3560 | 8 | 8364 | 1 | 1 | 0 |
| A-office | 20.0 | 115 | 0 | 3488 | 3903 | 21 | 11688 | 1 | 3 | 2 |
| C-office | 5.0 | 14 | 0 | 5653 | 5723 | 46 | 15698 | 1 | 3 | 3 |
