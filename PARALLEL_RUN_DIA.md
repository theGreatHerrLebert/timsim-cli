# Plan v4: parallelize `run_dia` (chunk-parallel, per measurement)

## MEASUREMENT RESULT (Gate 0 — done; it reversed the plan)

`perf` on a full toy dia-PASEF render (34,440 samples), self-time:

| stage | self% | notes |
|---|---|---|
| `dedup_and_quantise` | **34%** (+ ~7% SipHash = **~41%**) | per-frame HashMap dedup+quantize; SipHash on `(u32,u32)` keys dominates |
| sweep (`apply_transmission` + `pow`/sigmoid + `dia_render_range`) | **~24%** | |
| encode (`encode_frame_block` + `sort_dedup` + zstd) | **~12%** | |
| alloc churn | ~8% | |
| **sequential append** (`append_encoded_frame`) | **0.02%** | negligible — NOT a ceiling |

**This overturns the v3 "encode-only pipeline"** (below, kept for the record): keeping the sweep sequential
caps speedup at ~3× (Amdahl on the 24% sweep). But **append is negligible**, so parallelizing the WHOLE
per-frame render per chunk (sweep + dedup + encode; ~85% of runtime) yields **~7×** with a sequential
append. So the correct design is **chunk-parallel** (v2's shape), now justified by data, with the memory
discipline the reviews demanded.

## Step 1 (do first — independent single-threaded win): fix `dedup_and_quantise`

~41% of the render is SipHash on integer keys. Replace the default `HashMap<(u32,u32), f64>` with a
dep-free fast integer hasher (FxHash-style, packing `(scan,tof)` into a `u64` key). Expected ~1.5–1.7×
before any threading, near-zero risk. **Byte-identical** (only the hasher changes; sums per key and the
encoder's canonical re-sort are unchanged). Validate with the byte test, then re-profile.

## Step 2: chunk-parallel render (the ~7× on top)

Parallelize the per-frame render across the existing equal-ion frame chunks with **bounded in-flight**
concurrency (Codex's memory discipline): render+encode at most K chunks concurrently, **append each
chunk's blocks in frame order as it completes** (append is ~free), and **release** before admitting the
next — never a global `collect()`. `TimsTransmissionDIA` is `Sync` in pinned `mscore 0.5.0` (verified both
rounds: two read-only `HashMap`s + `f64`); still add `assert_sync` + source audit + repeated byte-compare.
Byte-identity holds via inclusive-overlap ion filter + global `m.order` (identical per-frame triple order)
+ `encode_frame_block`'s canonical re-sort + exact-`u128` `summed_intensities`.

---
## (superseded) v3: bounded encode pipeline — kept for the record

Rejected by the measurement: sweep is 24% and stays sequential here, capping at ~3×. Append being
negligible is what makes the fuller chunk-parallel design both correct and worthwhile.

## Keep, unchanged

- **Metadata pass** → `meta: HashMap<u64, IonMeta>` (~40 MB @ 617K; permanent O(precursors), acknowledged).
- **Streaming chunk loop** (equal-ion frame ranges, 512 MB-budgeted, `--render-chunks` override): each
  chunk streams `ion_spectra` once → builds its `Vec<DiaIon>` in global `m.order` → the sweep runs over
  `[fc0, fc1]`. This bounds **ion** memory exactly as today (one chunk resident). The truth's
  `with_ms2: HashSet<u64>` stays populated **here** in the sequential builder loop (no parallelism).
- `dia_render_range` (`src/ms2.rs`) unchanged: deterministic per-frame triple order (active set entered in
  `active_frames` start order, sorted by original index); calls back **only for non-empty frames**.

## The pipeline

Persistent across the whole render (spawned once, before the chunk loop; joined after):

```
[main thread: sequential sweep]                     produces frames IN ORDER
   for chunk: stream ions; dia_render_range(fc0,fc1, |frame, ms_type, triples| {
       tx.send(Job{ frame, ms_type, triples })   # BOUNDED sync_channel(depth D) → backpressure
   })
        │
        ▼  (bounded, depth D)
[N encode workers]  recv Job → (scans,tofs,ints)=dedup_and_quantise(triples,...) ;
                    blk = encode_frame_block(...) ;  done_tx.send(Done{ frame, ms_type, blk })
        │
        ▼
[writer thread]  reorder buffer BTreeMap<frame → (ms_type, blk)>; drain consecutively from
                 `next_frame`: write gap (empty) frames via gap_ms(f) up to it, then
                 append_encoded_frame(frame, frame*cycle_seconds, ms_type, blk); advance next_frame.
```

- **Bound**: producer blocks when the `sync_channel(D)` is full → at most `D` in-flight jobs (triples) +
  `N` encoding + reorder buffer `≤ D` encoded blocks. `D ≈ 2N` (N = `rayon::current_num_threads()` or a
  `--encode-threads`). **Independent of dataset size** and of `.d` size — a slow writer backpressures the
  sweep. This is the "bounded per-frame channel, not `collect()`ed chunk vectors" the review required.
- **Deps**: `std::sync::mpsc::sync_channel` (bounded) + `std::thread` — no new crate. (Or a `rayon::scope`
  with a bounded queue; std channels are simplest and already sufficient.)

## Byte-identity (why it holds — same set of `EncodedBlock`s, same order)

1. **Same triples per frame** as serial: the sweep is unchanged and sequential; each frame's triple order
   is identical to today. Chunk boundaries only split disjoint frame ID ranges (inclusive-overlap ion
   filter unchanged), so per-frame triples are identical.
2. **Per-frame encode is a pure function of its triples.** `dedup_and_quantise`'s randomized-`HashMap`
   drain order does NOT leak: `encode_frame_block` **re-sorts** to canonical `(scan,tof)` block bytes and
   computes `summed_intensities` as an exact `u128` sum (verified `ms-io/tdf_writer.rs` L213–217) — so the
   `EncodedBlock` is deterministic given the frame's triples, regardless of vector order or which worker
   ran it. (This is why round 2's "serial-vs-serial could differ" does not bite here.)
3. **Frames appended in frame order** by the reorder writer; gap/empty frames emitted with `gap_ms(f)`
   (= `sched.ms_level`-derived 0/9) exactly as the current serial loop. Result: identical `EncodedBlock`
   sequence → identical `.d` bytes.
4. **`DiaFrameMsMsInfo` schedule + writer config**: `set_dia_schedule(frame_to_group)` before finalize,
   `scan_mode: 9`, compression level, RT — all unchanged; only the producer→encoder→writer wiring changes.

Only compressed `EncodedBlock`s and owned triple `Vec`s cross thread boundaries (both `Send`); no shared
mutable state, no `Sync` requirement on `sched`/`Placement` (they stay on the sweep thread).

## Measurement (Gate 0, do first)

Instrument the current serial `run_dia` on the 617K dia-PASEF case with three coarse timers accumulated
across frames: **sweep** (`dia_render_range` deposit incl. transmission), **quantise+encode**
(`dedup_and_quantise` + `encode_frame_block`), **append** (`write_frame`). Report the split.
- If **encode ≫ sweep**: this pipeline yields ≈ min(N, encode/total) speedup — proceed.
- If **sweep ≫ encode**: pipeline won't help; reconsider (the transmission sweep would need parallelising,
  which reopens the `Sync` + memory questions). Decide before coding.
- Also confirm **append** isn't already the ceiling (sequential writer writes every compressed byte).

## Validation gates (with implementation)

- **Byte-identity**: `analysis.tdf_bin` AND `analysis.tdf` AND truth sidecar identical between the serial
  build and the pipeline, across several `--render-chunks` and `--encode-threads` values, incl.
  boundary-heavy synthetic ions. Plus **repeated serial** renders (rules out any latent nondeterminism).
- **Memory**: peak RSS stays ≈ `chunk-ion-budget + D·frame` on the 617K run and does **not** grow with the
  output `.d` size.
- **Speed**: wall-clock vs serial; and the per-stage split from Gate 0 recomputed on the pipeline.

## Effort / risk

One function (`run_dia`), ~70 lines: replace the inline `dedup_and_quantise+write_frame` callback with a
`tx.send`, add the worker + writer threads. Reuses `dia_render_range` / `dedup_and_quantise` /
`encode_frame_block` / `append_encoded_frame` / `gap_ms` verbatim. Risk is low (no `Sync`, no memory-model
change, encode already proven pure); the only real gate is Gate 0 confirming encode is the bottleneck.
