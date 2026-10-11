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
