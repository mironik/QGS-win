# Backward decoder plan for Sony High 4:2:2 10-bit

## Observed GOP

The 24-picture Sony sample is structurally regular:

- one decoded I anchor at POC 0;
- reference P anchors at POC 3, 6, 9, 12, 15, 18, and 21;
- two non-reference B pictures between consecutive anchors;
- each ordinary B picture references the surrounding P/I anchors in opposite L0/L1 order.

The bitstream already presents pictures in dependency-safe decode order. Parallel picture decoding is therefore neither required nor part of this plan.

## Work backward from the output

A finished picture is produced by this dependency chain:

```text
output bytes
  <- released decoded plane allocation
  <- deblocked reconstructed picture
  <- clipped prediction + inverse-transformed residual
  <- motion-compensated reference samples and residual coefficients
  <- resolved references, motion vectors, weights, QP and scaling
  <- CABAC syntax
  <- slice RBSP
```

The current decoder mixes CABAC, motion derivation, motion compensation, residual reconstruction, and metadata recording inside one large macroblock loop. That organization repeatedly crosses large code paths and writes prediction pixels to the frame before residual reconstruction reads and overwrites them.

## Target single-thread pipeline

### 1. Syntax and dependency stage

Decode CABAC into a compact frame plan in raster order. The plan contains structure-of-arrays metadata and compact reconstruction commands, not generic macroblock enums. It records:

- destination block or macroblock;
- resolved reference slots;
- motion vectors and implicit weights;
- interpolation class: integer, horizontal half, vertical half, diagonal half, or general quarter;
- residual presence, transform size, coefficient offset, QP and scaling selection;
- deblock metadata.

CABAC neighbor state and motion predictors remain raster ordered. No thread or speculative decoding is introduced.

### 2. Anchor reconstruction

The known I picture is reconstructed once and becomes the first immutable, deblocked anchor. Each P anchor is then reconstructed from the preceding anchors. The Sony GOP needs only a small rolling anchor set.

### 3. B-picture reconstruction

For the two B pictures between anchors, references are already final. Inter commands can be dispatched by concrete kernel rather than repeatedly matching generic syntax types.

Commands remain in destination order where required, but independent inter runs can use homogeneous loops:

- direct/skip 16x16;
- L0/L1 integer and half-pel 16x16;
- bi-pred integer and half-pel 16x16;
- split partitions;
- uncommon generic fallback.

This is batching within one thread, not parallelism.

### 4. Fused reconstruction store

The principal new kernel computes the final pre-deblock sample:

```text
pixel = clip10(prediction(reference, motion, weights) + transformed_residual)
```

Prediction is not first written to the destination and then read back by residual code. For zero-residual blocks the prediction kernel stores directly. For coded 4x4/8x8 blocks, prediction and transformed residual meet in a small local block or registers and are stored once.

### 5. Picture deblock and output

Deblocking remains a picture-level phase. It must not run early because intra prediction reads unfiltered neighboring samples. Once deblocking finishes, non-reference B planes transfer their byte allocation directly to output; reference anchors remain in the DPB until released.

## Why the rejected fused bi-pred experiment still matters

Fusing only the second predictor with averaging was slower because it added classification and duplicated filter dispatch while leaving the surrounding macroblock pipeline unchanged. The useful lesson is that fusion must cross the prediction/residual boundary and be selected once in the frame plan. Reclassifying inside every low-level tile cannot recover enough work.

## Implementation sequence

1. Introduce a `SonyFramePlan` and a compact `SonyInterCommand` without changing reconstruction.
2. Emit commands for dominant B skip/direct and P 16x16 paths while retaining the generic fallback.
3. Execute the plan and verify 24/24 exact output.
4. Add fused prediction-plus-residual kernels for 4x4 and 8x8 coded blocks.
5. Move interpolation classification out of inner tile functions into command creation.
6. Benchmark each completed pipeline boundary, while allowing an intermediate step to be neutral if it enables the next fusion step.

The acceptance condition is end-to-end single-thread frame throughput, not the isolated speed of an intermediate abstraction.

## Implemented fused boundary

Dominant coded full-16x16 B commands now cross the prediction/residual boundary in one local reconstruction operation. Luma uses a 16x16 local block and each 4:2:2 chroma plane uses an 8x16 local block. Motion interpolation writes those blocks, inverse-transformed residuals update them in place, and each completed plane is copied to the picture once. Split partitions continue through the generic exact fallback.

This removes the former prediction-to-picture followed by picture-read and residual rewrite for Y, Cb, and Cr. The Sony fixture remains 24/24 bit exact. Alternating single-core measurements against the luma-only fused implementation showed a median improvement from 24.31 to 24.70 fps, although system load still causes large outliers.

The CABAC input path now computes the renormalization shift once and fetches the required bits as a group instead of looping one bit at a time. Its byte cursor advances only when loading a byte; the externally visible bit position is derived on demand rather than updated for every consumed bit. Alternating single-core A/B runs under heavy system load showed 13.86 to 14.30 fps for the cursor change (about 3.2%); an earlier cleaner run measured 25.13 to 25.31 fps for batched renormalization alone. Both changes preserve 24/24 exact output.

Per-picture instrumentation changed the optimization priority. On the measured run the single I picture took about 125 ms before deblock, P pictures averaged about 54 ms, and B pictures about 36 ms. Deblocking remained about 8% of total picture time. Consequently, B-only dispatch and interpolation work cannot close the throughput gap; the common CABAC/residual path must shrink.

The first common-path cut removes aggregate diagnostic residual reports that were initialized, copied through I/P/B residual results, and immediately discarded after neighbor nonzero state had already been recorded. Alternating single-core A/B runs under the same heavy load improved the median from 20.85 to 21.83 fps (about 4.7%).

Residual block results now carry only a `u8` nonzero count instead of a multi-field diagnostic report with machine-word fields. The CABAC-produced raster coefficient arrays are transferred directly into 4x4 and 8x8 residual blocks using a checked bytemuck array cast, eliminating the second element-by-element raster copy. The report-to-count change was individually neutral in a clean A/B run; under a later heavily loaded run the `u8` layout and direct array transfer each showed higher medians, but those magnitudes require confirmation when the host is quiet. Correctness remains 24/24 exact.

## Shared-data dependency graph

The common basis is a decoded picture plan, not a partially written destination frame:

```text
CABAC syntax A
  = destination + block type + QP/scaling + references/motion + coefficients

A.references/motion  -> motion-compensated prediction --+
A.coefficients/QP    -> inverse quant/transform --------+-> clipped reconstruction
A.motion/nonzero/QP  -> deblock metadata -------------------> deblock after pixels exist
```

Motion compensation and inverse transform consume disjoint parts of A and neither consumes the other's result. On one superscalar core their independent instruction chains should be interleaved so reference-load latency can overlap transform arithmetic. They meet only at the final add/clip/store. This is instruction-level parallelism and software pipelining, not multiple threads. Deblock metadata can be derived alongside those chains, although filtering itself waits for reconstructed neighboring pixels.

CABAC within one slice remains a serial recurrence because every arithmetic-coder state depends on the previous bin. The useful overlap starts after each compact command has enough decoded state: prepare/prefetch reference data for command N+1 while transforming command N, then finish prediction and combine.

The implementation target changes accordingly:

1. Serial CABAC emits a compact structure-of-arrays picture plan.
2. Reference/motion commands, coefficient blocks and deblock metadata are stored in separate dense arrays keyed by destination.
3. A single-core software pipeline keeps multiple independent commands in flight: reference address/prefetch, interpolation, inverse transform, and final combine occupy separate dependency chains.
4. Transform batches process multiple independent coefficient blocks together so the CPU can issue operations from another block when one chain stalls.
5. The picture completes reconstruction before its deblock pass and DPB publication.

The strict single-thread target remains. No worker threads or frame-level parallel decode are introduced.

The first shared-input implementation prepares the 4x4 inverse-quant factors once for an inter luma/chroma reconstruction call. QP division, norm-table selection, shift direction and the 16 scaling products are then reused by all residual blocks in that call. This improved the alternating single-core median from 24.08 to 24.32 fps. Applying the same preparation unconditionally to sparse intra macroblocks regressed the median from 24.58 to 24.44 fps and was reverted; shared work must be activated only when enough nonzero AC blocks amortize it.

Intra reconstruction now activates that prepared state lazily on the second nonzero block, preserving the cheap one-block case while reusing shared factors thereafter. The alternating median improved from about 24.44 to 24.56 fps. Inter reconstruction likewise delays preparation until the first nonzero block; its measured median improved from about 24.22 to 24.50 fps.

CABAC context state and MPS are packed into one byte. Packing alone was end-to-end neutral, but precomputing the complete packed MPS/LPS transitions shortened the per-bin state recurrence and improved the alternating median from about 24.37 to 24.65 fps. This is retained because it attacks the serial dependency chain rather than adding concurrency.

A direct zero-residual store experiment bypassed construction and inspection of zero residual arrays, but its median was indistinguishable from the lazy inter baseline (about 24.24 fps). It was reverted. Zero-residual final storage is therefore not the present bottleneck; the next structural target remains interleaving genuinely coded transform work with motion-compensation load/filter work.

The coded 4x4 path had another ordering defect: it classified a block as DC-only only after inverse-quantizing all 16 coefficients. DC classification now gates inverse quantization, so a DC-only block evaluates only coefficient (0,0) and immediately applies its constant residual. Under a heavily loaded host the paired single-core median improved from about 21.02 to 21.52 fps (about 2.4%); the Sony GOP remains 24/24 exact. A separate direct-strided transform experiment regressed about 1.5% because it lost optimization over fixed 4x4 arrays, so that experiment was reverted.

The first OpenH264-derived change replaces the one-byte CABAC reader with a 32-bit reservoir held in a 64-bit register. Reads refill to at least 32 available bits and renormalization consumes directly from that register; byte alignment is computed from the logical bit position so prefetched bytes remain valid. Seven alternating single-core runs improved the median from 25.30 to 25.76 fps (about 1.8%), with 24/24 exact Sony output. This reservoir is the foundation for moving AC dequantization into coefficient decode without adding another intermediate pass.

The common CABAC MPS path now returns directly when its reduced range is already normalized, matching OpenH264's fast-path ordering instead of entering the general renormalization helper on every decision. Seven alternating single-core runs improved the median from about 25.37 to 25.73 fps (about 1.4%).

Inter-luma 4x4 AC coefficients are now dequantized at CABAC placement time. Only positions present in the decoded significant list are multiplied; reconstruction consumes explicitly marked predequantized blocks and cannot scale them twice. The first implementation regressed because it rebuilt the 16-factor table per macroblock. Caching the prepared table in the residual decoder and refreshing it only when QP changes turned the result positive: under heavy host load the seven-run paired median improved from about 19.93 to 20.69 fps (about 3.8%). Intra, chroma, 8x8 and DC/Hadamard paths remain unchanged until equivalent specialized representations exist.

## Whole-pipeline isolation

A temporary syntax-only diagnostic kept CABAC, motion-vector derivation, coefficient decoding, neighbor state and deblock metadata intact while suppressing inter-picture pixel reconstruction. It was removed after measurement. P pictures fell from roughly 45--54 ms to 24--34 ms before deblock, so prediction/reconstruction accounts for about 40% of their cost. B pictures fell from roughly 30--40 ms to 10--17 ms, making bi-prediction/reconstruction about 60% of their cost. I pictures remained near 97 ms because their reconstruction was intentionally not suppressed. End-to-end time fell from about 1.09 s to 0.65 s even though I reconstruction and the complete deblock pass remained.

This rejects two overly broad assumptions: neither CABAC alone nor transform alone dominates the whole GOP. A trial SSE4.1 vertical half of the 4x4 inverse transform retained 24/24 exact output but measured 24.69 versus 24.71 fps and was removed. The next high-leverage target is therefore the B-picture bi-prediction dataflow: fuse or interleave the two reference interpolation chains and final weighting, rather than optimizing an isolated transform.

Motion-pattern counting then exposed a much denser special case than expected: across the benchmark's warm-up and measured pass, 23,993 bi-predicted 16x16 luma blocks used integer positions in both references. An isolated AVX2 backend now handles the equal-weight integer case with 16-sample loads and the native rounded unsigned average instruction; the same routine handles the 8-sample 4:2:2 chroma rows. The scalar/portable weighted path remains the fallback. Eleven reverse-order paired runs measured a median new/old ratio of 1.008 (about 0.8%), with 24/24 exact output. This is retained, but it also shows that integer blending itself is not the full B-picture cost; fractional interpolation and repeated command/plane dispatch are the next targets.

A follow-up integer-plus-half-pel fusion removed one temporary predictor and combined the remaining filtered predictor directly with the integer reference. Although exact, eleven reverse-order paired runs produced a 0.991 median ratio (about 0.9% slower), so it was removed. The regular two-buffer loops evidently schedule/vectorize better than the more conditional fused loop. Future fractional work must use a dedicated straight-line kernel selected before entering the pixel loop, rather than branching on reference order inside it.

A dedicated SSE4.1 horizontal half-pel kernel then moved selection fully outside the pixel loop and performed six-tap filtering plus the equal-weight bi-average directly in registers. It too remained neutral (median ratio 0.997 across eleven reverse-order pairs) and was removed. This narrows the problem further: isolated interpolation arithmetic is already efficient enough; meaningful improvement requires collapsing the repeated per-plane/per-command setup around the complete B reconstruction.

Profiling the dominant integer-luma group showed that 23,441 calls also had integer chroma positions and equal 32/32 weights; only 1,008 calls had a chroma half-position. A whole-4:2:2 shortcut therefore bypassed the generic luma/chroma entry points and directly blended Y, Cb and Cr after one classification. Despite the high hit rate, eleven reverse-order pairs measured a 0.996 median ratio, so the shortcut was removed. Bounds/origin setup is not expensive enough to justify duplicating the established plane kernels. The remaining B reconstruction cost is chiefly memory traffic and the required prediction work itself, not dispatch overhead.

## 2026-10-12 audit: coded inter commands dominate

The current 24-picture Sony GOP decodes exactly against FFmpeg (24/24 frames). Five single-core runs of the current release benchmark measured 0.787-0.802 s, median 0.792 s (30.3 fps); this is a same-build baseline, not an A/B speedup claim. An instrumented run attributed about 89 ms to the measured I picture, 261 ms to seven P pictures, and 390 ms to sixteen B pictures. With temporary per-macroblock timing, reconstruction accounted for 44.7-46.5% of P decode time and 58.6-60.8% of B decode time across three runs. The timing probes were removed after this audit.

The delayed `SonySlicePlan` is reached by only 6,786 of 73,440 P macroblocks (9.2%) and 53,397 of 155,040 B macroblocks (34.4%). Among the measured B macroblocks, 24,726 had luma residual without chroma and 3,540 had chroma without luma. Most remaining coded B macroblocks require both planes. A change confined to sorting or batching the zero-residual plan therefore cannot deliver the required whole-GOP speedup. The dominant path is immediate coded B reconstruction in `SonyBInterCommand::execute_fused`, followed by coded P reconstruction and the serial CABAC/motion syntax path.

The next structural experiment should keep CABAC and neighbor state in raster order, but prepare inverse-transformed residual tiles independently of motion-compensated reference loads inside each coded inter command. Schedule the two independent chains before their final add/clip/store and avoid a second destination-frame pass. Compare against an identical source baseline with alternating single-core runs and require 24/24 exact frames. A prior syntax-only diagnostic took about 0.65 s for this GOP, so pixel work alone cannot reach the 0.48 s needed for 50 fps; CABAC and motion derivation must also improve.

A same-source CABAC experiment removed the local copies and write-back in `read_renorm_bits`. It passed all 130 tests and 24/24 exact Sony frames, but a second reverse-order eight-pair benchmark won only three pairs and had a median old/new speed ratio of 0.986. The change was reverted; local state copies are not the useful CABAC bottleneck.

Two further coded-path experiments were rejected. Branchless coefficient classification (`is_zero` and `is_dc_only`) won only four of eight paired runs with an effectively 1.000 old/new median ratio. Passing CABAC's nonzero/DC-only masks directly to fused inter reconstruction passed all 130 tests and 24/24 exact Sony frames; however, its first eight paired runs showed a 1.072 median ratio while the next twelve showed 0.978, with only six wins in the latter. Host timing drift is larger than the apparent gain, so neither experiment is retained. The mask experiment also enlarged the residual command structure. The next candidate should reduce an actual full-frame or per-command memory traversal, not just replace local coefficient scans with metadata.

An additional whole-GOP test replaced the scalar 8x16 chroma bi-prediction blend with the existing eight-sample vector blend. It remained FFmpeg-exact (24/24) and passed 130 tests, but won five of twelve alternating pairs with median old/new ratio 0.995; it was reverted. The prior temporary syntax-only diagnostic remains the stronger directional evidence: about 0.65 s with inter pixel reconstruction suppressed, already above the 0.48 s 50-fps target. A fresh baseline profile measured 0.869 s for the GOP, with 93.5 ms in I, 289.6 ms in seven P, and 433.7 ms in sixteen B pictures (picture decode only). The next structural design must reduce both serial syntax/motion derivation and coded inter pixel work; optimizing one chroma or luma arithmetic loop cannot close the roughly 0.39 s gap.

On 2026-10-13, direct in-place addition of predequantized 4x4 luma residuals removed temporary prediction/reconstruction block copies and stayed exact against FFmpeg, but won six of twelve paired runs (median old/new 1.002); it was reverted. Const-generic specialization of the CABAC residual path for chroma-DC and dequantized inter-luma also passed all 130 tests and 24/24 frames, but won seven of sixteen pairs (median 0.998); it too was reverted. Neither local data movement nor per-coefficient category selection is currently a demonstrated whole-GOP bottleneck.

The architecture boundary is now explicit in the live path: CABAC and motion syntax are decoded in raster order; only zero-residual full-block commands enter `SonySlicePlan`; coded P/B commands immediately run motion-compensated prediction, residual reconstruction and frame writes before CABAC advances. A useful next experiment must change that boundary: queue **coded** inter commands in bounded raster-order segments, flush before any intra prediction or other pixel dependency, and execute a single-core prediction/transform/combine pipeline over each segment. Merely extending the existing zero-residual sort without a dependency barrier would risk incorrect intra neighbors, while queueing the entire picture would enlarge memory traffic. The experiment needs a same-source baseline, exact FFmpeg comparison and paired whole-GOP timing before adoption.

That bounded-segment experiment was run with up to 64 full-16x16 coded P/B commands per segment, flushing before intra reconstruction and using the existing plan's grouped single-core executor. It passed 130 tests and matched FFmpeg exactly on all 24 Sony GOP frames. However, two reverse-order 16-pair measurements gave old/new medians 1.010 and 0.986 (nine and seven wins respectively); the combined result is effectively neutral, with occasional substantial regressions. The code was reverted. This rejects **materializing and sorting** coded commands with the current arenas, not the dependency observation: the extra writes/reads of coefficient arrays and buffered predictions appear to cancel any scheduling benefit. The next candidate should overlap prediction loads and coefficient transforms *within a command* or a tiny fixed-size pipeline without storing a whole segment of residuals.

The CABAC nonzero-neighbor state was also tested as a two-macroblock-row ring instead of picture-sized arrays. The slice cursor is raster-ordered, and each current slot was cleared before decoding its macroblock; a row-reuse unit test and all 24 Sony frames passed exactly. Two 16-pair measurements showed only 1.005 and 1.003 old/new median ratios (nine and ten wins), below the threshold for retaining a state-lifetime change. It was reverted. This rules out full-picture neighbor-state cache footprint as the major source of the 50-fps gap on this clip.

A temporary CABAC residual timer showed four slices per Sony picture. For the timed 24-picture GOP, the existing 4x4 inter-luma residual decoder consumed about 9 ms and chroma residual decoding about 54 ms out of roughly 858 ms end-to-end. The timer was removed. Windows Performance Recorder's CPU profile could not start on this host (`0xc5585011`, system-performance profiling policy), so no sampling attribution was inferred. All 32 pictures seen including warmup had four slices with `disable_deblocking_filter_idc = 1`. Skipping allocation/recording of the whole deblock grid in that case remained neutral across sixteen reverse-order pairs (median old/new 0.999, seven wins), and was reverted. The larger remaining cost lies outside those measured residual subcalls and deblock metadata.

The four slice starts are exactly `[0, 2040, 4080, 6120]` in every measured I/P/B picture. This gives four independent CABAC arithmetic recurrences per picture, but a release microbenchmark of four decoder states on one core found no reliable benefit from scalar round-robin interleaving relative to finishing each state in sequence; the diagnostic example was removed. This does not test SIMD lane decoding of real syntax, but it does not justify converting the full parser to a four-lane state machine merely for scalar interleaving.

Temporary timing of the immediate, coded full-16x16 B fused path measured 68,871 calls across the timed GOP: about 91 ms for motion-compensated prediction, 31 ms for luma reconstruction and 22 ms for chroma reconstruction. The sixteen B pictures together took about 432 ms in that run, so this fused path accounts for roughly one third of B-picture decode time. B direct-motion derivation was measured separately at about 17 ms across 81,052 calls; optimizing that helper alone cannot close the gap. All timing probes were removed. The remaining B cost is distributed across CABAC/motion syntax, zero-residual plan execution and non-full-block prediction; the next audit must measure those coarse paths before another specialized arithmetic kernel is attempted.

That coarse B audit measured roughly 44,000 deferred B plan commands at 47 ms, and 16,824 non-full-block B fallback reconstructions at 58 ms, in separate profiled runs. Taken with the 144 ms coded full-block fused path, this accounts for about 249 ms of approximately 432 ms B-picture decode time; the residual ~180 ms includes CABAC syntax, motion derivation and other setup. These timings come from different instrumented runs and are approximate, not additive cycle attribution. A temporary FFmpeg 8.1.1 run on the same GOP with decoder `-threads 1` and raw YUV422P10 output took about 0.51-0.64 s wall unpinned, but about 0.63-0.72 s total process CPU time. Pinning the FFmpeg process to one logical processor gave about 0.64-0.80 s wall including startup and I/O. Therefore FFmpeg's unpinned wall fps is not a fair proof that 50 fps is attainable with one core; its auxiliary pipeline can still overlap. All temporary QGS timers were removed.

A further B-path experiment bypassed the generic eight-block residual loop for each chroma plane when the coded full-16x16 command had no chroma residual, storing the already computed prediction directly. It passed all 130 tests and stayed exact on all 24 FFmpeg reference frames. Ten alternating single-core pairs gave six trial wins, but host contention swung individual runs from roughly one to more than six seconds; a follow-up process-CPU-time comparison also drifted strongly and split four wins each. There is no defensible speedup claim, so the change was reverted. This reinforces that skipping a small zero-block scan does not address the dominant B prediction and serial syntax costs. The next experiment needs a more stable measurement environment and a structural reduction in full-block prediction or syntax work.

The bypass was then reconsidered as a dependency for simplifying the complete B-plane path, not as an isolated arithmetic speedup. The coded full-16x16 B executor now uses the same direct stores already used by the P executor for absent luma **and** chroma residuals, removing creation and traversal of synthetic zero blocks on both planes. All 130 tests and 24/24 exact Sony frames still pass. Eight alternating pairs on one core were split four wins each, with a median trial/base time ratio near 1.01 under host load; no throughput gain is claimed. The small change is retained because it gives the next plane-scheduling experiment a clean distinction between prediction-only and prediction-plus-transform paths, without needing to enter a generic residual loop for absent data. End-to-end performance remains the acceptance criterion for the later combined change.

The next combined experiment completed B luma prediction/reconstruction/write before starting chroma prediction, aiming to reduce simultaneous live buffers and cache pressure. It preserved all 130 tests and 24/24 FFmpeg-exact frames, but required a second kernel dispatch and reference resolution. Ten alternating whole-GOP pairs on one core won only four times; the median trial/base time ratio was about 1.08, with a substantial slow outlier. The scheduling change was reverted while the absent-residual direct stores were kept. Thus plane-at-a-time execution with duplicate dispatch is not the right shape; a future interleaving attempt must resolve references and classify the kernel once, then preserve that prepared state across plane stages without duplicating checks.

The prepared-reference variant classified L0/L1/Bi once, resolved the frame references once, and reused that prepared kernel for separate luma and chroma stages. It was again FFmpeg-exact on 24/24 frames, but won only two of eight alternating single-core pairs against the direct-store baseline; median trial/base time was about 1.06. The extra prepared-kernel value and second dispatch still cost more than the reduced buffer lifetime saves. This variant was reverted. The next candidate must fuse useful transform work into prediction without splitting the prediction kernel into two dispatches; or shift focus to the serial CABAC/syntax floor.

The next CABAC common-path experiment combines `range_lps` and both packed-context transitions into one 32-bit table entry indexed by `(packed_context, range_quadrant)`. Each arithmetic bin now loads one prepared entry rather than three separate tables, shortening a shared serial dependency without changing context semantics. A test covers all 128 packed states and four quadrants; the full 130-test suite and 24/24 FFmpeg comparison passed before the added table test. Three alternating one-core whole-GOP series gave 7/10, 7/10, and 6/12 trial wins. Their median trial/base time ratios were approximately 0.944, 0.822, and 0.976, but the middle series had extreme host-load outliers and the last series was near neutral. The combined 20/32 wins suggest possible benefit, not a reliable percentage speedup. Retain provisionally as a shared-path simplification and remeasure under a quieter host; the original transition and range tables remain the construction source, so the state machine is unchanged.

With the host reported steadier (roughly 32% overall CPU, antivirus over 13%), the same saved pre-table binary and rebuilt table binary were retested alternately on one logical processor. On affinity mask 1, 20 pairs split exactly 10/10 and the median trial/base time ratio was 0.999; all paired checksums matched. On affinity mask 128, 12 reverse-order pairs gave 7 trial wins and median ratio 0.969, but the range still included large scheduling outliers. The table is **not a demonstrated throughput improvement**; the first core's stable ~0.79-0.82 s runs show the 30-fps baseline remains intact. Retain the table only as a shared-path representation for subsequent combined experiments, not as an accepted stand-alone speedup. Revert it if a later combined change does not exploit it or if cleaner measurements show a regression.

To address the historical 30.3-fps comparison directly, clean Git HEAD `454d52f` (the source corresponding to that earlier baseline) was rebuilt and alternated with the current working-tree decoder on the same logical CPU. Twenty pairs split 10/10; the median paired new/old time ratio was 0.999, with standalone medians 0.813 s old and 0.811 s new for 24 frames. Checksums matched in every pair. The historical 0.792-s run was achieved under lighter conditions; neither version reproduced it consistently during this A/B. There is no evidence of a code regression from this paired test, but also no proven overall speedup. The current release example was rebuilt after the comparison.

## Encoder-informed Sony slice constraints (2026-10-13)

The 24-frame Sony test GOP has 96 slices: 4 I, 28 P, and 64 B. Every slice signals one active reference per list, `weighted_bipred_idc = 0`, `cabac_init_idc = 0`, and `disable_deblocking_filter_idc = 1`; B slices use spatial direct prediction. This is a measured property of this clip, not a general XAVC guarantee. H.264's encoder may choose these coding tools while the decoder must obey the bitstream. A guarded B fast path used constant reference indices and default 32/32 weights, and whole-picture deblock metadata allocation/recording was skipped when every slice disabled filtering. Both changes remained exact (131 tests and 24/24 FFmpeg frames). Against a saved pre-experiment binary on one logical processor, the combined version won 7/16 alternating pairs and had median trial/base time ratio 1.003; the changes were reverted. The baseline's existing `num_ref_idx_active_minus1 == 0` path already avoids CABAC reference-index bins and disabled deblock already avoids the filter pass. Removing their residual bookkeeping cannot bridge the gap to 50 fps. Future encoder-informed work should target the actual B prediction and coded macroblock dataflow, or a sizeable serial syntax dependency, with an end-to-end paired test.

Removing the zero-residual plan's in-place sort was also tested. This left the command batches in decode order and preserved exact output on all 24 Sony frames, but sixteen paired one-core runs split 8/8 with median trial/base ratio 0.999. The sort was restored. Its extra traversal is not a measurable whole-GOP bottleneck; the useful target remains execution of pixel prediction and the serial syntax path, not rearrangement of already decoded commands.

The release benchmark now accepts an optional in-process GOP repeat count. Each repeat uses a fresh decoder, preserving the same stream semantics while excluding process startup and input parsing from the measured loop. Eight repeats pinned to one logical processor produced 192 frames in 6.250 s, or 30.72 fps, with median GOP time 0.782 s. This is a better stable baseline for future whole-pipeline experiments, not a speedup. The output conversion path is also now understood: a reference picture still needed by the DPB is cloned into an owned output plane, while a released picture transfers its byte buffer without a copy on little-endian hosts. Eliminating the former would require an output ownership/API change and needs separate timing before it can be claimed as a high-leverage target.

Temporary picture-level timers on eight in-process repetitions measured the steady GOP at roughly 740-751 ms inside `decode_picture_into_frame`, 6-7 ms in access-unit parsing, and 9.5-10.5 ms cloning seven still-referenced output pictures. The first reported repetition also included the eight-frame warmup in the diagnostic counters and was excluded from these steady figures. Output moves were negligible. The probes were removed. This rules out output ownership/copying as a route to 50 fps: even eliminating all seven copies would save only about 1.3% of current GOP time. The next high-leverage change must reduce work inside picture decoding itself.

The coded full-16x16 B path was then compared against its existing generic alternative by temporarily forcing those commands through `copy_from_references` followed by `apply_sony_b_residual`. Both paths produced 24/24 FFmpeg-exact frames. With four GOP repetitions per process and eight alternating single-core pairs, the generic path won only 2/8 and its median general/fused time ratio was 1.021. The trial was removed. The current fused local-prediction path is a real whole-pipeline improvement over a two-pass destination-frame reconstruction; future work should reduce work *within* the fused path rather than split it back into prediction and residual passes.

The motion-field representation was tested next. Full-16x16 P/B motion normally writes the same `MotionSample` into sixteen 4x4 slots per list, so a temporary uniform-per-macroblock marker stored one sample and materialized the full row only if later partial writes occurred. Both neighbor lookup and colocated lookup honored the marker; 131 tests and 24/24 Sony frames passed. Eight alternating one-core pairs with four GOP repetitions per process split 4/4 and had median trial/base ratio 1.002. The extra condition on frequent neighbor reads appears to cancel saved writes. The representation was reverted. Further motion work should target fewer neighbor derivations or a different compact layout without an extra unpredictable per-read branch, rather than only suppressing duplicate stores.

As a diagnostic upper bound for CABAC error-check overhead, `decode_decision` temporarily returned decoded bins without checking its sticky end-of-slice fault. This deliberately weakens malformed-input handling and was never a candidate for retention. Eight alternating one-core pairs with four GOP repetitions each produced only 3/8 trial wins and median trial/base ratio 1.001. The original fault check was restored. Moving that check to syntax-element boundaries cannot plausibly yield a material whole-GOP gain, while it would complicate error semantics.

A same-core latency-overlap test issued L2 prefetches for the luma reference rows of coded `B_Pred_16x16` macroblocks immediately after motion and CBP syntax became known, before residual CABAC decoding. The output stayed 24/24 exact. Eight alternating pairs with four GOP repetitions each split 4/4, median trial/base ratio 0.995, with several sizeable losses. The prefetches were removed: advancing memory requests alone does not establish a useful whole-GOP gain.

The next experiment must replace a *complete* coded B-macroblock dataflow, not add another isolated helper. Candidate: decode a bounded pair of full-16x16 coded B macroblocks into fixed-size on-stack descriptors, perform inverse transforms into signed residual tiles independent of their motion-compensated prediction, then combine reference prediction plus signed residual into the destination in one store. Flush before any intra/pixel dependency and before slice end. This avoids the 64-command arena/sort regression, retains single-core execution, and exposes independent transform/reference chains for scheduling without an extra full-frame pass. A signed inverse-transform API is required so negative residuals are not clipped before prediction; residual arithmetic must match H.264's final clipping exactly. Before implementation, benchmark the isolated transform-to-signed-tile plus one-store kernel against the existing fused command on representative B blocks. Stop if it fails to beat the current 0.78-0.82 s whole-GOP baseline reproducibly, and always require 24/24 FFmpeg-exact output and all library tests.

The signed-luma-tile candidate was implemented for coded B 4x4 luma blocks. It computed signed inverse-transform results before motion-compensated prediction, then combined and clipped during one destination-frame write. Both the original pixel-wise combiner and a revised 4x4-block combiner passed 131 tests and matched FFmpeg on all 24 Sony frames. The first had 2/8 wins and median trial/base ratio 1.010 under uneven host load. The cleaner second comparison had only 1/8 wins and median ratio 1.014. The full prototype was reverted. The independent signed tile added a 1 KiB write/read per coded macroblock plus classification and destination scatter; on one core, exposing independence this way costs more than the scheduling benefit. Do not expand this representation to chroma. A future fused kernel would need to keep transformed values in registers or combine them at 4x4 granularity without materializing a 16x16 signed tile; otherwise the existing fused predictor/reconstruction remains faster.

Four-way SIMD across independent 4x4 luma inverse transforms was tested next, using four 32-bit lanes for the same coefficient position in four blocks. Zero and DC-only blocks retained exact scalar semantics, and mixed-pattern plus randomized SIMD/scalar tests passed (133 tests total); Sony output was 24/24 FFmpeg-exact. The first two one-core series against the same saved scalar baseline had 6/8 and 7/12 SIMD wins, with median trial/base ratios 0.994 and 0.984. A third eight-pair series under changed host load had only 2/8 wins and median ratio 1.060. Requiring at least two AC blocks before invoking SIMD also lost against the one-AC threshold (3/8 wins, median ratio 1.018). The implementation was reverted: the overall speedup is not robust, and four-way gathering/scattering plus zero/DC classification adds substantial code and work around a relatively small transform fraction. A better SIMD design would need coefficients laid out in four-lane form directly by residual decoding, avoiding per-macroblock transposition; such a change must be evaluated together with CABAC and reconstruction, not as a stand-alone transform replacement.
