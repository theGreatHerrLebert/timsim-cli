Reading additional input from stdin...
OpenAI Codex v0.144.6
--------
workdir: /scratch/timsim-demo/timsim-cli
model: gpt-5.6-terra
provider: openai
approval: never
sandbox: read-only
reasoning effort: medium
reasoning summaries: none
session id: 019fef63-47bb-72d2-a039-7a3938bd374e
--------
user
Review this design plan as an independent engineer who cares about scientific reproducibility and artifact provenance.

CONTEXT: this is a mass-spectrometry DIA simulator (Rust). It writes vendor-format instrument files (.d) plus 'answer key' parquet files used to score search engines. It currently stamps three keys describing the chromatographic peak shape into the artifact, so a copied/restored artifact can answer 'which kernel made you?' without re-running. The render is about to change from ONE global peak shape to a PER-PEPTIDE shape drawn from Beta distributions, which breaks the current stamp's key property (a total round-trip: the stamp reconstructs an object that compares == to what the render used).

FOCUS ON:
1) Is the proposed 'generative descriptor + realized summary + input binding' actually sufficient to answer the stale-artifact question? What can go wrong that this misses? Be specific about failure scenarios.
2) The plan drops emg_k in per-peptide mode rather than writing a population mean into it, arguing a reader could not tell a mean from a value. Is that the right call?
3) The input binding (peptide_rt_sha256): is a file hash right, or should it be a content hash of the two draw columns? What breaks if the upstream artifact is regenerated bit-differently but semantically identically?
4) Answer the 7 open questions at the end, briefly, where you have a view. Question 1 is the most important: is trading self-containment for correctness right?
5) Anything the plan does not consider at all. Especially: version/migration hazards, multi-writer consistency (the same stamp goes into a SQLite table AND parquet schema metadata for 4 different writers), and whether summary statistics can be made to lie.

Be concrete and specific. Distinguish 'this is wrong' from 'this is a taste call'. Cap ~800 words.

<stdin>
# Plan — provenance for a per-peptide peak shape

> **STATUS: PROPOSED, not implemented.** Written 2026-08-11, before wiring the per-peptide draws,
> deliberately: the stamp has to be decided *before* eleven call sites are changed, not after.
>
> Companion to [`PEAK_SHAPE.md`](PEAK_SHAPE.md), which describes the shape work this invalidates.

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

This is a *total* description of the map. Given `peptide_rt` and these keys, every peptide's shape is
recomputable exactly — which restores the original property one level up: not "reconstruct the
`PeakShape`" but "reconstruct **every** `PeakShape`".

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

### C. Input binding — which draws these came from

| key | why |
|---|---|
| `peptide_rt_sha256` | binds the artifact to the exact upstream draws |

Without this, A + B describe a map whose inputs cannot be identified, and the "is this stale?"
question is only half-answered. With it, a `.d` handed to a collaborator states both the law and the
draw set.

### Backward compatibility

Keep `peak_shape` with values `emg` / `gaussian` / **`per-peptide`**, so an old reader that only
understands the name still gets a truthful answer rather than a missing key. **Drop `emg_k` in
per-peptide mode rather than writing a mean into it** — a reader that finds `emg_k = 0.476` has no
way to know it is a population mean and not the run's `k`, and that misreading is worse than an
absent key. `parse_shape` must therefore fail loudly, not fall back, when it sees `per-peptide`.

The global mode (`--emg-k` explicitly set, or no per-peptide draws available) keeps today's three
keys unchanged and its exact round trip.

## Open questions for review

1. **Is a generative descriptor an acceptable substitute for a total round trip?** It restores
   reproducibility only in conjunction with `peptide_rt`. Today's stamp is self-contained. Is
   trading self-containment for correctness the right call, or should the `.d` carry something
   stronger?
2. **Is `peptide_rt_sha256` the right binding**, given the flow may relocate or regenerate that
   artifact? Is a content hash of the two draw columns better than a file hash?
3. **Should `--emg-k` explicitly set disable the per-peptide draws** (as `--cycle-seconds` explicit
   overrides the reference), or compose with them (e.g. scale them)? Overriding is simpler and
   matches local convention; composing is more expressive and probably nobody wants it.
4. **What should a render do when `peptide_rt` lacks the draw columns** — an artifact predating
   them? Refuse, or fall back to the global constant and stamp `global/v1`? Falling back is
   friendlier; refusing makes the stale-artifact question answerable. This codebase has generally
   chosen refusing.
5. **Is the realized summary worth the pass**, or is it redundant given the truth parquet carries
   per-peptide rows? It is the only part that catches a wiring bug *in the artifact itself*.
6. **Does anything already read these three keys** that a schema change would break? Known readers:
   `parse_shape`, `tests/golden_gaussian.rs` (asserts the `.d` and answer key self-identify), and
   the golden manifest's `parent_differs_only_by` list.
7. **`n_sigma` stays global** — correct? It is a truncation radius, not a shape parameter, and v1
   truncates by cumulative probability (`target_p = 0.999`) rather than by a sigma count at all.
   That is a separate v1 divergence this plan does not address.

## What this plan does NOT cover

* The mobility axis. Per-ion `ccs_std` will need the same treatment, and the design there is a
  **fixed calibration gain** rather than v1's draw-set-dependent population mean-pinning — measured
  at 1.4–4.1% swing across subsets, which would make an ion's width depend on which other ions are
  in the run. That is a separate document.
* v1's cumulative-probability truncation (question 7).
* The necroflow fingerprint problem: none of `--sigma-frames`, `--peak-shape`, `--emg-k`,
  `--cycle-seconds` appear in the command string, so cached renders do not rebuild when these
  defaults change. That is a flow-side fix and is why this stamp exists at all.
</stdin>
codex
The direction is sound, but the proposal is not yet sufficient for a strong stale-artifact answer. The central issue is that a descriptor plus a hash proves only what it precisely binds—not necessarily what the renderer actually consumed.

1. Generative descriptor + summary + binding

This is the right replacement for a single-object round trip, provided the binding identifies the *effective draw map*, not merely an upstream file. As written, it misses several failure modes:

- The renderer may join draws to peptides differently than assumed: sequence normalization, I/L handling, modified-peptide identity, duplicate sequences, missing rows, or a changed salt/key derivation. The descriptor can be correct while a different map is consumed.
- The implementation can change while retaining the same law name: rounding/casting order, Beta decoding precision, frame conversion, EMG implementation, `k == 0` criterion, or truncation behavior. `per-peptide/v1` needs a strict compatibility contract or implementation/version digest.
- The run may shape a subset of upstream peptides. Hashing the whole input does not say which rows were used.
- A renderer bug can use global `k` or the wrong columns yet produce plausible min/mean/max.

So: add a canonical `effective_draws_sha256` over the actual identity→`sigma_hat`,`k_hat` map used by the render, after canonicalization and selection. Include identity encoding, column types/units, null policy, duplicate policy, and a deterministic sort. Also record an upstream blob/file digest if operational provenance matters; it answers “which exact supplied artifact?” while the semantic digest answers “which values did I use?”

For strongest reproducibility, publish the upstream draw artifact as a content-addressed companion and record its identifier/URI. If offline self-containment is a requirement, embed the effective draw table (or a compact canonical sidecar) in the deliverable. A hash alone detects mismatch; it does not enable reconstruction.

2. Dropping `emg_k`

Correct. Writing a mean in the legacy scalar key is actively misleading. Absence plus `peak_shape=per-peptide` and a loudly failing legacy parser is better. This is not merely taste: a scalar `emg_k` no longer has the legacy semantic type.

3. File hash versus content hash

Use both, with distinct names. A file hash is useful chain-of-custody evidence but is too fragile as the semantic binding: Parquet rewriting, row-group layout, compression, metadata, column order, or regenerated dictionary encodings can change bytes while preserving draws. Conversely, a semantic hash alone loses provenance of the exact source artifact.

Canonical content hashing must decide whether row order is meaningful. It normally should not be for an identity-keyed draw map; sort canonically and reject conflicting duplicate identities. Do not hash “two columns” alone—include the peptide identity column, otherwise the mapping is unbound.

4. Open questions

1. Trading self-containment for correctness is right only if the artifact explicitly becomes a manifest referring to a durable, content-addressed draw set. If collaborators must reconstruct from a lone `.d`, embed or package the draw map. Do not claim self-contained reproduction otherwise.
2. Content hash of the effective map is the primary binding; retain source-file hash as secondary provenance.
3. An explicit `--emg-k` should disable per-peptide mode, provided the artifact records the effective mode and override decision. Silent scaling/composition is a new model and needs its own descriptor.
4. Refuse missing draw columns by default. Permit fallback only through an explicit compatibility option and stamp `global/v1`; never silently select it.
5. Keep a realized summary, but define it as an invariant computed from the exact effective map/render inputs, not incidental ion-pass state.
6. Treat reader discovery as an implementation gate: search all consumers, fixtures, manifests, and external schema docs. Add a compatibility test for every writer/reader pair.
7. `n_sigma` is currently misleading if v1 actually uses cumulative-probability truncation. This is a correctness problem, not a separate cosmetic issue. Record `truncation_law` and its parameter(s), and only retain `n_sigma` where it actually governs rendering.

5. Missing operational safeguards

Use one canonical, versioned provenance record—e.g. canonical JSON plus `provenance_sha256`—as the source for SQLite and all four Parquet writers. Each destination writes the identical serialized record/digest; readers validate agreement and fail on divergence. SQLite writes should be transactional; Parquet outputs need a completion manifest so partial/mixed writer outputs are detectable.

Define migration rules now: versioned keys, required/forbidden-key sets per `elution_model`, explicit legacy interpretation, and no “missing means Gaussian” defaults.

Finally, summaries can lie through wrong population definition (unique peptides versus ions), duplicated peptides, NaNs, precision loss, stale accumulators, or values computed before later overrides. Record count semantics and add stronger checks: a canonical effective-map digest, exact zero/nonfinite counts, and perhaps deterministic quantiles. Min/mean/max alone are easy to match accidentally—or deliberately.
tokens used
6,953
The direction is sound, but the proposal is not yet sufficient for a strong stale-artifact answer. The central issue is that a descriptor plus a hash proves only what it precisely binds—not necessarily what the renderer actually consumed.

1. Generative descriptor + summary + binding

This is the right replacement for a single-object round trip, provided the binding identifies the *effective draw map*, not merely an upstream file. As written, it misses several failure modes:

- The renderer may join draws to peptides differently than assumed: sequence normalization, I/L handling, modified-peptide identity, duplicate sequences, missing rows, or a changed salt/key derivation. The descriptor can be correct while a different map is consumed.
- The implementation can change while retaining the same law name: rounding/casting order, Beta decoding precision, frame conversion, EMG implementation, `k == 0` criterion, or truncation behavior. `per-peptide/v1` needs a strict compatibility contract or implementation/version digest.
- The run may shape a subset of upstream peptides. Hashing the whole input does not say which rows were used.
- A renderer bug can use global `k` or the wrong columns yet produce plausible min/mean/max.

So: add a canonical `effective_draws_sha256` over the actual identity→`sigma_hat`,`k_hat` map used by the render, after canonicalization and selection. Include identity encoding, column types/units, null policy, duplicate policy, and a deterministic sort. Also record an upstream blob/file digest if operational provenance matters; it answers “which exact supplied artifact?” while the semantic digest answers “which values did I use?”

For strongest reproducibility, publish the upstream draw artifact as a content-addressed companion and record its identifier/URI. If offline self-containment is a requirement, embed the effective draw table (or a compact canonical sidecar) in the deliverable. A hash alone detects mismatch; it does not enable reconstruction.

2. Dropping `emg_k`

Correct. Writing a mean in the legacy scalar key is actively misleading. Absence plus `peak_shape=per-peptide` and a loudly failing legacy parser is better. This is not merely taste: a scalar `emg_k` no longer has the legacy semantic type.

3. File hash versus content hash

Use both, with distinct names. A file hash is useful chain-of-custody evidence but is too fragile as the semantic binding: Parquet rewriting, row-group layout, compression, metadata, column order, or regenerated dictionary encodings can change bytes while preserving draws. Conversely, a semantic hash alone loses provenance of the exact source artifact.

Canonical content hashing must decide whether row order is meaningful. It normally should not be for an identity-keyed draw map; sort canonically and reject conflicting duplicate identities. Do not hash “two columns” alone—include the peptide identity column, otherwise the mapping is unbound.

4. Open questions

1. Trading self-containment for correctness is right only if the artifact explicitly becomes a manifest referring to a durable, content-addressed draw set. If collaborators must reconstruct from a lone `.d`, embed or package the draw map. Do not claim self-contained reproduction otherwise.
2. Content hash of the effective map is the primary binding; retain source-file hash as secondary provenance.
3. An explicit `--emg-k` should disable per-peptide mode, provided the artifact records the effective mode and override decision. Silent scaling/composition is a new model and needs its own descriptor.
4. Refuse missing draw columns by default. Permit fallback only through an explicit compatibility option and stamp `global/v1`; never silently select it.
5. Keep a realized summary, but define it as an invariant computed from the exact effective map/render inputs, not incidental ion-pass state.
6. Treat reader discovery as an implementation gate: search all consumers, fixtures, manifests, and external schema docs. Add a compatibility test for every writer/reader pair.
7. `n_sigma` is currently misleading if v1 actually uses cumulative-probability truncation. This is a correctness problem, not a separate cosmetic issue. Record `truncation_law` and its parameter(s), and only retain `n_sigma` where it actually governs rendering.

5. Missing operational safeguards

Use one canonical, versioned provenance record—e.g. canonical JSON plus `provenance_sha256`—as the source for SQLite and all four Parquet writers. Each destination writes the identical serialized record/digest; readers validate agreement and fail on divergence. SQLite writes should be transactional; Parquet outputs need a completion manifest so partial/mixed writer outputs are detectable.

Define migration rules now: versioned keys, required/forbidden-key sets per `elution_model`, explicit legacy interpretation, and no “missing means Gaussian” defaults.

Finally, summaries can lie through wrong population definition (unique peptides versus ions), duplicated peptides, NaNs, precision loss, stale accumulators, or values computed before later overrides. Record count semantics and add stronger checks: a canonical effective-map digest, exact zero/nonfinite counts, and perhaps deterministic quantiles. Min/mean/max alone are easy to match accidentally—or deliberately.
