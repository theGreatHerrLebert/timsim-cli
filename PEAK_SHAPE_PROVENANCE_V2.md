# Plan — provenance for a per-peptide peak shape

> **STATUS: PROPOSED, revised after review.** Written 2026-08-11 before wiring the per-peptide
> draws, deliberately: the stamp has to be decided *before* eleven call sites are changed, not after.
> Revised the same day against
> [`PEAK_SHAPE_PROVENANCE_V2.codex-review.md`](PEAK_SHAPE_PROVENANCE_V2.codex-review.md).
>
> Companion to [`PEAK_SHAPE.md`](PEAK_SHAPE.md), which describes the shape work this invalidates.
>
> **The review's central correction, accepted:** the first draft called section A "a total
> description of the map". It is not, and the word *total* was doing the same overclaiming as an
> earlier "agree by construction" that the same reviewer struck down. A + B + C **detects some stale
> artifacts; it does not prove realization.** What actually reached the renderer depends on things no
> proposed field records — the identity-key algorithm and its salt, the Beta draw semantics, the
> zero-`k` threshold, frame rounding, peptide selection and dedup, and the renderer version itself.
> Every one of those can change the rendered shapes while leaving all of A unchanged.
>
> The plan below is restructured around that: a **versioned, canonical provenance contract** with a
> realized-set digest, rather than a descriptor that describes intent and hopes it was honoured.

## The problem in one line

The render is about to stop having *a* peak shape, and the artifact currently stamps one.

## What the stamp is today, and the property that breaks

`src/provenance.rs` writes three keys into `analysis.tdf`'s `GlobalMetadata` and into every answer
key's Arrow schema:

| key | value |
|---|---|
| `peak_shape` | `gaussian` \| `emg` |
| `emg_k` | the tailing factor; `0` for the Gaussian |
| `n_sigma` | the truncation radius |

Its load-bearing property is stated in the module doc, and it is stronger than "we wrote down the
shape":

> Those three are exactly the inputs to `PeakShape::emg`, which is what makes the round trip
> **total**: `parse_shape` reconstructs a `PeakShape` that compares `==` to the one the render used,
> derived constants and all. Recording only the shape NAME would be a label; recording the
> constructor arguments is a reproduction.

That exists because of a real defect: `truth.parquet` was byte-identical between a Gaussian render
and an EMG render, so **a stale artifact could not be identified as stale from its own contents**. A
`.d` that is copied, re-pathed, or handed to a collaborator arrives with no fingerprint attached.

**Per-peptide shapes break the round trip, not the motivation.** After wiring, a run contains
thousands of distinct `k` values (one per peptide, `k = 10·k_hat`, `k_hat ~ Beta(1,20)`) and
thousands of distinct widths (`sigma_hat ~ Beta(4,4)` mapped onto the gradient band). There is no
single `PeakShape` to reconstruct. Writing one anyway is the exact failure this codebase has now
fixed twice — a wrong kernel certifying itself as correct (`54dea2f`: `PeakShape::emg` silently
downgrading to Gaussian and stamping `gaussian, 0.0`; `c714675`: a bare `k` whose `0` case had to be
remembered by every caller).

So the question is not *whether* to change the stamp. It is what replaces a reproduction when
reproduction of a single object is no longer meaningful.

## What the render will actually be parameterised by

After the wiring, the elution shape of peptide *p* is fully determined by:

```
sigma_frames(p) = [lo + sigma_hat(p)·(hi − lo)] / cycle_seconds
                  where (lo, hi) = 0.75·mid, 1.25·mid
                        mid      = gradient_seconds/3600·0.75 + 1.125
k(p)            = 10 · k_hat(p)
shape(p)        = Emg(k(p), n_sigma)  if k(p) > 0 else Gaussian
```

with `sigma_hat(p)`, `k_hat(p)` read from `peptide_rt` — **identity-keyed on the peptide sequence**,
`blake2b(sequence#salt)`, so they are a property of the peptide and not of the run.

That matters for the design: the per-peptide draws are *already* durably recorded in an upstream
artifact. The render does not invent them. So the stamp does not need to carry per-peptide values —
it needs to carry **the map from those draws to this run's shapes**, plus enough to detect that the
upstream artifact is the one that was used.

## Proposal

Replace the three scalar keys with a **generative descriptor** plus a **realized summary** plus an
**input binding**. Three groups, three different jobs.

### A. Generative parameters — what the shapes were drawn from

| key | example | why |
|---|---|---|
| `elution_model` | `per-peptide/v1` | the scheme; distinguishes from `global/v1` below |
| `sigma_law` | `gradient-affine/v1` | names the law, so a future law is not confused for this one |
| `sigma_band_seconds` | `[1.13437, 1.89062]` | the resolved band — the two numbers the draw maps into |
| `gradient_seconds` | `1861.323203` | the input to the band |
| `cycle_seconds` | `0.105445749226364` | seconds→frames; the value the clock fix now inherits |
| `sigma_hat_dist` | `beta(4,4)` | the draw's distribution, for a distributional check |
| `k_upper` | `10.0` | `k = k_hat·k_upper` |
| `k_hat_dist` | `beta(1,20)` | as above |
| `n_sigma` | `3.0` | unchanged; still global |

plus, per the review, the fields without which "recompute every shape" is not actually possible:

| key | why it is REQUIRED, not nice-to-have |
|---|---|
| `provenance_schema_version` | unknown MAJOR must **fail closed**. Without it there is no migration story at all. |
| `elution_model_version` | the implementation, not the law: the zero-`k` threshold, frame rounding and the EMG's own constants live here. A change to any of them moves the shapes while every other field is unchanged. |
| `identity_key` | e.g. `blake2b-64/sequence#salt`, WITH the salt. The draws are keyed by this; a different normalisation silently repoints every peptide to a different draw. |
| `truncation_law` + its parameter | see Q7 — `n_sigma` is the v2 rule, and describing it as though it governed v1 (which truncates at `target_p = 0.999`) is a mis-statement, not a simplification. |

Given `peptide_rt` **and** these keys, every peptide's shape is recomputable — which moves the
original property up a level: not "reconstruct the `PeakShape`" but "reconstruct **every**
`PeakShape`". This is a weaker guarantee than today's self-contained round trip, and the plan says so
rather than implying otherwise.

### B. Realized summary — what actually came out

Cheap to compute during the ion build pass, and the only part that can catch a wiring bug:

| key | example |
|---|---|
| `sigma_frames_mean` / `_min` / `_max` | `14.34` / `10.76` / `17.93` |
| `emg_k_mean` / `_min` / `_max` | `0.476` / `0.0` / `6.21` |
| `n_peptides_shaped` | `12228` |
| `n_gaussian_collapsed` | `3` (peptides whose `k_hat` drew to 0) |

Deliberately summary statistics, not a histogram: this is a provenance record, not a dataset. The
distributional comparison against v1 reads the *truth parquet*, which has per-peptide rows.

**Six aggregates are easy to lie with, so they are not the commitment.** The review's improvement,
adopted: also stamp a **deterministic digest over the actually-rendered set**

```
realized_shape_digest = sha256( for each rendered peptide, in identity order:
                                 identity_key || sigma_frames || k || truncation )
```

which is one key, is as compact as a single aggregate, and cannot be satisfied by a writer that
computed its summaries from the wrong population — the failure the aggregates cannot catch. The
aggregates stay as a **human-readable diagnostic**; the digest is the commitment.

Invariants a verifier must check, all of them cheap: `min <= mean <= max`, every value finite,
`n_peptides_shaped` equal to the population actually rendered (not the candidate set), and every
statistic computed **after** final unit conversion and rounding rather than before.

### C. Input binding — which draws these came from

| key | why |
|---|---|
| `peptide_rt_draws_content_sha256` | **canonical CONTENT hash** over `(identity_key, rt_sigma_hat, rt_k_hat)` — the semantic binding |
| `peptide_rt_file_sha256` | optional, byte-level audit trail |

A file hash alone was wrong, and the review is right about why: parquet regenerated with different
row grouping, compression, column order or producer version has different bytes and **identical
draws**, so a file hash reports a false mismatch on a semantically identical input. The content hash
must include the **identity key**, not just the two draw columns — hashing the draws alone cannot
detect draws bound to the wrong peptides, which is the more dangerous error.

Canonical encoding has to be pinned explicitly (schema version, normalised sequence bytes, null
policy, IEEE-754 representation with a canonical NaN, deterministic row order) or the hash is not
reproducible across producers, which defeats its purpose.

Without this group, A + B describe a map whose inputs cannot be identified, and "is this stale?" is
only half-answered.

### Backward compatibility

Keep `peak_shape` with values `emg` / `gaussian` / **`per-peptide`**, so an old reader that only
understands the name still gets a truthful answer rather than a missing key. **Drop `emg_k` in
per-peptide mode rather than writing a mean into it** — a reader that finds `emg_k = 0.476` has no
way to know it is a population mean and not the run's `k`, and that misreading is worse than an
absent key. `parse_shape` must therefore fail loudly, not fall back, when it sees `per-peptide`.

The global mode (`--emg-k` explicitly set, or no per-peptide draws available) keeps today's three
keys unchanged and its exact round trip.

## Multi-writer consistency and migration — the largest gap in the first draft

The review's blunt framing, accepted: *"SQLite metadata plus four Parquet writers is a distributed
schema update, not a simple key rename."*

* **One typed provenance record, one canonical serialisation.** Every writer consumes the same
  constructed value; none recomputes summaries or digests independently. Today `provenance.rs`
  already centralises the three keys, but a per-writer recomputation of section B would let the four
  writers disagree about the same run.
* **A post-write verifier** that reads back every output and asserts the provenance payload is
  canonically equal across all of them. Without it, "all writers stamp it" is an intention.
* **Atomicity.** No artifact may survive with `analysis.tdf` stamped and the answer key unstamped,
  or vice versa — that combination is indistinguishable from tampering after the fact.
* **Migration policy.** Recognise legacy global stamps; reject *mixed or ambiguous* stamps; never
  infer per-peptide provenance from an artifact that does not assert it.
* **Fail closed on unknown major versions**, in every reader.

## Provenance is detection, not prevention

The first draft listed the necroflow fingerprint problem under "not covered" and left it there. That
understates the relationship, and the review is right to push:

> provenance detects a bad cached result after the fact; it does not prevent one.

So the fingerprint must itself include the descriptor version, the resolved parameters, the
draw-content hash and the renderer version. The stamp makes a bad artifact *identifiable*; only the
fingerprint stops it being *produced and reused*. Neither substitutes for the other, and shipping the
stamp alone would leave the original cache bug fully intact while feeling like it had been addressed.

## Resolved questions (answers from review, adopted)

1. **Trade self-containment for correctness — yes**, provided the binding is content-addressed and
   versioned. But do not *describe* the result as self-contained. If offline reconstruction matters
   for collaborators, ship a sidecar manifest or embed the draw columns in the answer-key package.
2. **Content hash, with identity included; keep the file hash as an optional audit trail.**
3. **Explicit `--emg-k` disables per-peptide mode**, stamped unambiguously. Composition is a
   different model and would need its own explicit flag.
4. **Refuse missing draw columns by default.** A silent global fallback turns an incompatible
   upstream artifact into a plausible-but-different simulation — the exact failure mode this whole
   module exists to make visible. An explicit legacy flag only if operationally forced.
5. **Keep the realized summary, but demote it**: a diagnostic, not evidence. The digest is the
   commitment, and both must be computed from the exact final shaped-peptide set.
6. **Repository-wide reader and fixture audit before changing anything**, and specify behaviour for
   absent, legacy, and conflicting keys — external consumers and old artifacts both exist.
7. **`n_sigma` as stamped is mis-described.** Record `truncation_law` and its parameter. v1 truncates
   at a cumulative probability (`target_p = 0.999`), not a sigma count, so stamping `n_sigma` as
   though it were the governing rule states something untrue about the v1 comparison.

## Remaining open questions

1. **Is the realized-shape digest affordable at scale?** It is one pass over the shaped peptides at
   render time, but it forces a deterministic identity order over a set that is currently built as a
   `HashMap`. At 9M precursors / ~1-2M peptides that is a sort, not a scan.
2. **Where does the digest live for the Thermo and SCIEX writers?** They are upstream crates with no
   provenance seam, which is why the answer key carries the stamp today. A post-write verifier that
   asserts cross-writer equality needs a value it can read back from all four.
3. **Does `peptide_rt` need to carry its own content hash**, computed by `timsim-rt`, so the render
   does not have to re-hash a large artifact on every run?

## What this plan does NOT cover

* The mobility axis. Per-ion `ccs_std` will need the same treatment, and the design there is a
  **fixed calibration gain** rather than v1's draw-set-dependent population mean-pinning — measured
  at 1.4–4.1% swing across subsets, which would make an ion's width depend on which other ions are
  in the run. That is a separate document.
* v1's cumulative-probability truncation (question 7).
* The necroflow fingerprint problem: none of `--sigma-frames`, `--peak-shape`, `--emg-k`,
  `--cycle-seconds` appear in the command string, so cached renders do not rebuild when these
  defaults change. That is a flow-side fix and is why this stamp exists at all.
