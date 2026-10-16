### Extreme FFT log

Written by Claude (an AI model by Anthropic), October 2026. The steps from the AI FFT ([fft/fft_ai.odin](../fft/fft_ai.odin)) to the extreme FFT ([fft/fft_extreme.odin](../fft/fft_extreme.odin)), with what each one measured, the ones that made it slower too.

Apple M1 Pro, Odin `dev-2026-10:84bc3fc21`, `-o:speed -microarch:native`. Real input, N = 4096 unless noted, microseconds per transform, the fastest of 5 runs. Runs of the same code differ by up to 2–3%.

```
odin run benchmark_ai -o:speed -microarch:native       # whole transforms, against pffft
odin run benchmark_stages -o:speed -microarch:native   # every pass on its own
```


#### Start

| | µs |
|---|---|
| radix-2 FFT (fft.odin) | 25.0 |
| AI FFT | 3.45 |
| AI FFT, fused multiply-adds in the complex multiply | 3.18 |
| pffft ordered | 2.55 |
| pffft unordered | 2.30 |

Odin doesn't fuse a multiply and an add on its own, `simd.fma` asks for it.


#### Radix-8

All stages radix-8, one radix-4 or radix-2 stage last.

| | µs |
|---|---|
| radix-8, twiddles held for the whole inner loop, 13 vector spills to the stack | 3.33 |
| radix-8, twiddles loaded in the inner loop, 2 spills | 3.43 |

One radix-8 pass cost 0.63–0.74 µs, two radix-4 passes 0.62. Fewer spills didn't help either. Dropped.


#### Passes of the AI FFT

| pass | µs |
|---|---|
| first, stride 1 | 0.62 |
| radix-4, stride 4 | 0.51 |
| radix-4, stride 16 | 0.35 |
| radix-4, stride 64 | 0.32 |
| radix-4, stride 256 | 0.31 |
| radix-2, stride 1024 | 0.24 |
| split | 0.86 |

A radix-4 pass is 128 iterations of 28 vector operations, about 3,600, and the M1 does 4 a cycle: 0.28 µs. The middle passes are close to that. The first pass, the stride 4 pass and the split aren't.


#### Split

| | µs |
|---|---|
| bin k and bin N/2 − k from the same loads (still radix-8) | 3.07 |
| back to radix-4, the last stage run inside the split | 2.84 |

The split and the last stage together: 0.76 µs, against 0.24 + 0.86 apart. 8192: 1.67 against 0.62 + 1.75.


#### Twiddles and the inner loop

| | µs |
|---|---|
| twiddles stored in all four lanes, p = 0 without multiplies, twiddles loaded for every q | 3.15 |
| the same, twiddles loaded once for each p | 2.83 |
| the p = 0 test out of the inner loop, the inner loop unrolled by two, twiddles passed as an array | 3.03 |
| back to the AI FFT's inner loop, p = 0 in a loop of its own, twiddles as six separate vectors | 2.75 |

The unrolled version slowed the middle passes to 0.49, 0.46 and 0.42 µs from 0.35, 0.32 and 0.31. The `[6]` array of twiddles, passed into an inlined procedure, is the likely cause. The last row has them at 0.34, 0.30 and 0.27, p = 0 without multiplies saving most in the late passes.


#### Instructions

| | µs |
|---|---|
| complex multiply as `fma(-x_im, w_im, x_re * w_re)`: 16 negates down to 5, folded into 11 `fmls` | 2.70 |
| first pass loads and stores in the shape LLVM turns into `ld2` and `st4` | 2.67–2.73 |

`ld2` and `st4` made no measurable difference, the first pass stayed at 0.60 µs.


#### Stride 4

At stride 4 the inner loop runs once for each p, six broadcast twiddle loads for eight data loads.

| | stride 4 pass | whole |
|---|---|---|
| before | 0.51 | 2.70 |
| twiddles of four p in six vector loads, each p's taken from its lane | 0.32 | 2.56–2.63 |
| the same at stride 16, with a loop over q | 1.13 (stride 16: 0.52) | 3.03–3.06 |

The stride 16 version was reverted.


#### Split, second look

Instructions per iteration of each loop, from `-build-mode:asm`:

| loop | before | after |
|---|---|---|
| first pass | 52 | 52 |
| radix-4 pass | 49 | 48 |
| radix-2 last stage + split | about 60 | 47 |
| radix-4 last stage + split | 244 | 191 |

The split loops reloaded the twiddle pointer and `plan.half` from the plan on every iteration, the radix-4 one the plan pointer from the stack too, 18 loads. The compiler can't rule out the stores through `out` changing the plan. Taken into locals before the loop.

The other changes to the split:

* The mirrored bins turned around in the shuffle that pairs re with im, `(3, 7, 2, 6)` and `(1, 5, 0, 4)`, instead of a reverse of each (`rev64` and `ext`) and then the pairing.
* `-odd_re` computed directly, the ½ in two multiply-adds, no negate.
* The split twiddles only up to N/4: the split pairs k with N/2 − k, k never goes past N/4. Half the table.
* The scalar edge bins by mask and shift instead of `%` and `/`.

| | 1024 | 4096 | 8192 |
|---|---|---|---|
| last stage + split, before | 0.187 | 0.712 | 1.594 |
| last stage + split, after | 0.165 | 0.623 | 1.441 |
| whole, after | 0.539–0.555 | 2.44–2.45 | 5.37–5.47 |
| pffft ordered, same runs | 0.550–0.562 | 2.51–2.54 | 5.77–5.83 |


#### Now

| N | radix-2 | AI FFT | extreme FFT | pffft ordered | pffft unordered |
|---|---|---|---|---|---|
| 1024 | 5.27 | 0.715 | 0.539–0.555 | 0.550–0.562 | 0.487 |
| 4096 | 24.2 | 3.12 | 2.44–2.45 | 2.51–2.54 | 2.33 |
| 8192 | 52.1 | 6.58 | 5.37–5.47 | 5.77–5.83 | 5.22 |

The extreme FFT's output is in bin order like pffft ordered's. pffft's unordered output is in its own order, only useful for a convolution or a correlation and back.

| pass, 4096 | AI FFT | extreme FFT |
|---|---|---|
| first, stride 1 | 0.602 | 0.603 |
| radix-4, stride 4 | 0.494 | 0.320 |
| radix-4, stride 16 | 0.338 | 0.321 |
| radix-4, stride 64 | 0.309 | 0.291 |
| radix-4, stride 256 | 0.301 | 0.266 |
| last stage and split | 0.227 + 0.838 | 0.623 |
| whole | 3.138 | 2.427 |

Left: the first pass, 52 instructions an iteration like a middle pass's 48, takes twice as long. Its `ld2` and `st4`, or the zips and plain stores before them, cost about the same.
