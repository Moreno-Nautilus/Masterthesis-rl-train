# HIL-SERL on the Franka FR3 — training cost per insert

Real-robot HIL-SERL (RLPD, frozen ResNet-10 encoder, ZED-mini wrist camera, PS4 shared
autonomy). Every policy trained from scratch from 20 human demos plus online interventions.

## Results

| Insert | Task | Approach | Demos | Demo ep. length | Online transitions | Learner steps | Final intervention | Autonomous eval |
|---|---|---|---|---|---|---|---|---|
| 0 | plumbers screw | vertical −Z | 20 | —¹ | 29 001 | 136 000 | 0 % | **15/15** |
| 1 | pb_pipe | horizontal −Y | 20 | 230 | 44 500 | 192 000 | 0 % | **15/15** |
| 2 | cooling base | vertical −Z | 20 | 67 | 11 500 | 67 000 | 0 % | **15/15**² |
| 3 | cooling base (2nd part) | vertical −Z | 20 | 77 | **9 001** | **50 000** | 0 % | **15/15**² |
| 4 | *(pending)* | | | | | | | |

**All four policies were evaluated hands-off at 15 trajectories each and seated the part
15/15 times at 0 % intervention.** These supersede the earlier figures (10/10 for inserts 0–1,
and the per-500-step training-dump counts for inserts 2–3), which were taken either at a
smaller eval count or read off training dumps rather than a dedicated eval run.

¹ Insert 0's demo `.pkl` was not kept, so its demo episode length is unrecoverable. Its
transition count and 0 % final intervention were recovered from the buffer dumps after the
fact; the run itself predates the per-dump logging used for inserts 1–3. Final-episode
lengths ran 70–113 steps.

² Inserts 2 and 3 were *also* tracked via training dumps during the run (15 and 12 successes
per 500 steps at 0 % intervention respectively); the 15/15 above is the later dedicated
hands-off eval.

## The headline finding

**Demo episode length predicts training cost.** The mean length of the 20 human
demonstrations is a cheap, up-front predictor of how much online data the policy will need:

| | insert 1 | insert 2 | insert 3 |
|---|---|---|---|
| demo episode length | 230 steps | 67 steps | 77 steps |
| online transitions to converge | 44 500 | 11 500 | 9 001 |
| learner steps | 192 000 | 67 000 | 50 000 |

Inserts 2 and 3 have near-identical demo lengths (67 / 77) and near-identical training
cost (11.5k / 9.0k transitions); insert 1, at 3× the demo length, cost ~4× as much.

A **vertical descent with orientation locked** is a far easier problem than a **horizontal
insert that needs continuous counter-steer**. Insert 1's difficulty is not incidental: with
the part held horizontally the wrist cannot hold attitude during the approach (a measured
several degrees of error), so the operator has to correct continuously, and orientation must
stay in the action space. Insert 2 pins orientation straight down and never touches it.

## Robustness: moving the fixture

Level 1 of the robustness test — the cooling-base fixture was moved **276 mm** to a new
table spot (x +84, y +262, z +3) and, because it was re-placed by hand, ended up at a yaw
**~90° from training**. Only `RESET_POSE` and the safety box were updated; the policy,
guards, reset noise and orientation lock were untouched (`EnvConfigInsert2Moved`).

**Zero-shot: the policy was completely lost.** It never seated the part.

Fine-tuning it back (warm start from `checkpoint_67000`) vs the original from-scratch run:

| transitions | fine-tune @ new spot | insert 2 from scratch |
|---|---|---|
| 2 500 | 2 succ / 12 % interv. | 4 succ / 60 % |
| 5 000 | 2 succ / **47 %** | 10 succ / **0 %** |
| 6 000 | 6 succ / 10 % | 9 succ / 0 % |
| 6 500 | 6 succ / **0 %** | — |
| 7 000 | 13 succ / 0 % | — |
| 7 500 | 13 succ / 0 % | — |
| **converged** | **7 500** | **11 500** |

Final state of the fine-tune: 67 successes over 7 501 transitions, **27 consecutive
successful episodes**, last 12/12, three closing dumps at 0 % intervention, 37-step
episodes (`checkpoint_109000`). That matches the from-scratch endpoint (13 vs 15 successes
per 500 steps, 37 vs 32 step episodes), so it is a genuine recovery, not a partial one.

**The warm start is not free.** For the first ~5 000 transitions the fine-tune went
*backwards* — intervention climbed from 12 % to 47 % — while the from-scratch run was
already at 0 %. The pre-trained policy does not start from ignorance, it starts from
confident wrongness: a visual mapping anchored on a scene rotated 90° from the current one,
with a critic assigning high value to actions that are now wrong. It has to unlearn that
before it can relearn. Only after ~5 000 transitions did it break through — and then it
went from 2 to 13 successes per 500 steps within 1 500 transitions.

Net: **7 500 vs 11 500 transitions**, so fine-tuning won by ~35 % — but it looked like a
failure for two thirds of the run, and a warm start naively promises far more than that.

`RelativeFrame` absorbs the *translation* (proprioception is reset-relative, so the new xy
is invisible to the state branch) but not the *rotation*: the wrist camera is rigidly
mounted, so a 90° yaw change rotates every image relative to training. A translation-only
test would isolate the two — not yet run.

## Convergence shape

Both inserts show the same pattern — a long flat phase, then a sharp climb:

- **Insert 1**: flat at ~1 success / 500 steps for the first **25 000** transitions, then
  climbed from ~27 000 to 9 successes / 500 at 0 % intervention.
- **Insert 2**: 66 % intervention at step 500 → 25 % by 3 000 → **0 % by 5 000**, then
  roughly **doubled** its success rate again between 6 000 and 11 500 (9 → 15 per 500).

**Do not stop on a flat stretch, and do not stop at the first 0 %-intervention dump.**
Both runs were called converged prematurely and both improved substantially afterwards.

## Signals to watch (in order of usefulness)

1. **Intervention rate falling** — the real progress signal; measured from the *demo_buffer*
   dump sizes, not the online buffer.
2. **Episode length falling** — the policy is reaching the goal faster.
3. **Successes per 500-step dump rising** — noisiest of the three; single dumps swing wildly.

A converged policy beats the human demos on episode length (insert 2: 32.5 steps vs 67).
