# Run config comparison (hardened-env runs; broken-env dropped)

## BEFORE holiday (< Jul 3)

| run | date | res | env | maxep | fc | lr_sched | lr | kl_thr | torque | aux | seed | result |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| vision_fuse_1 | 06-25 | 160 | 64 | 500 | 128 | adaptive | 0.0001 | 0.008 | n/a | - | 0 | 95.5% |
| state_hardened_1 | 06-25 | state | 128 | 1000 | - | adaptive | 0.0001 | 0.008 | n/a | - | 0 | 66.2% |
| vision_hardened_1 | 06-26 | 160 | 64 | 1000 | 128 | adaptive | 0.0001 | 0.008 | n/a | - | 0 | 75.6% |
| blank_hardened_1 | 06-26 | 160 | 64 | 300 | 128 | adaptive | 0.0001 | 0.008 | n/a | - | 0 | 33.8% |
| vision_appearance_1 | 06-27 | 160 | 64 | 3000 | 128 | adaptive | 0.0001 | 0.008 | n/a | - | 42 | 72.7% |
| vision_auxhead_1 | 06-30 | 160 | 64 | 2000 | 128 | adaptive | 0.0001 | 0.008 | n/a | Y | 42 | 64.1% |
| vision_holehead_1 | 06-30 | 160 | 64 | 1500 | 128 | adaptive | 0.0001 | 0.008 | n/a | Y | 42 | 68.6% |
| vision_res224_1 | 07-01 | 224 | 48 | 1200 | 128 | adaptive | 0.0001 | 0.008 | n/a | - | 42 | 69.7% |
| vision_torque_1 | 07-02 | 224 | 48 | 1200 | 128 | adaptive | 0.0001 | 0.008 | ON | - | 42 | 68.8% |

## HOLIDAY chain & after (>= Jul 3)

| run | date | res | env | maxep | fc | lr_sched | lr | kl_thr | torque | aux | seed | result |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| vision_res256_1 | 07-03 | 256 | 32 | 1200 | 128 | adaptive | 0.0001 | 0.008 | off | - | 42 | 27.3% |
| base48 | 07-04 | 224 | 48 | 3500 | 128 | adaptive | 0.0001 | 0.008 | off | - | 42 | 40.4% |
| base48_s7 | 07-05 | 224 | 48 | 3500 | 128 | adaptive | 0.0001 | 0.008 | off | - | 7 | 36.5% |
| fc256 | 07-06 | 224 | 48 | 3500 | 256 | adaptive | 0.0001 | 0.008 | off | - | 42 | 37.9% |
| fc512 | 07-08 | 224 | 48 | 3500 | 512 | adaptive | 0.0001 | 0.008 | off | - | 42 | 48.4% |
| lstm2048 | 07-09 | 224 | 48 | 3500 | 128 | adaptive | 0.0001 | 0.008 | off | - | 42 | 30.1% |
| lstm512 | 07-09 | 224 | 48 | 3500 | 128 | adaptive | 0.0001 | 0.008 | off | - | 42 | - |
| baseline_fc512 | 07-28 | 224 | 48 | 1500 | 512 | adaptive | 0.0001 | 0.016 | off | - | 42 | 0.2% |
| baseline_fc512_kl008 | 07-29 | 224 | 48 | 1500 | 512 | adaptive | 0.0001 | 0.008 | off | - | 42 | 48.6% |
| smoke_reboot | 07-29 | 224 | 48 | 150 | 128 | adaptive | 0.0001 | 0.008 | off | - | 42 | - |

**Note:** lr_schedule/lr/kl_threshold are identical across all runs (adaptive / 1e-4 / 0.008) except the two flagged experiments. `torque` = n/a means pre-Jul-1 code (field didn't exist). So differences are hyperparameters, not code.
