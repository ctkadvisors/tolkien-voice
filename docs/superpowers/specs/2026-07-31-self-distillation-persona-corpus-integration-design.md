# Self-distillation corpus integration into train-mvp-v6-persona.sh

Date: 2026-07-31

Status: Proposed for implementation

Repository: `ctkadvisors/tolkien-voice` (private)

## Context

The reward-filtered self-distillation pipeline (merged, live-verified 2026-07-31)
produces a valid corpus of real, live-captured audio examples --
`scripts/build-self-distillation-corpus.py`'s output: individual WAV +
alignment-JSON pairs plus a flat `manifests/train.jsonl`
(`{"path": ..., "duration": ...}` rows). A full run against the live
PersonaPlex instance on Spark produced 20 examples (aragorn + sauron,
2026-07-31).

`scripts/train-mvp-v6-persona.sh` cannot consume this corpus as-is.
Reading `src/tolkien_voice/training/persona_manifest.py` (the step that
builds the per-character weighted manifest the trainer actually reads)
shows it never touches audio or the source manifest's own content
directly -- it filters and reweights *purely through
`DialogueBlueprint` metadata* (`character`, `split`, `curriculum`,
`timing`), then looks up the matching row in a source train/eval
manifest by `blueprint.id` (`_rows_by_id`, matching on the audio
filename's stem). Without a corresponding blueprint entry, a manifest
row is invisible to `persona_manifest.py` regardless of what's in the
source `train.jsonl` -- appending self-distillation rows directly to
the v6 corpus's existing `manifests/train.jsonl` would do nothing.

## Decision: synthesize a minimal `DialogueBlueprint` per self-distillation example

Confirmed with the project owner (2026-07-31): build a real, valid
`DialogueBlueprint` for each self-distillation `RenderedExample`,
tagged `timing=TimingKind.INTERRUPTION` -- every self-distillation
example is, by construction, an interruption-response instance, so this
is not a stretch, it's the accurate classification -- which means these
examples automatically receive `persona_manifest.py`'s existing
interruption-reweighting boost (up to 3x representation, or 2x under
its "low diversity" fallback) with zero new weighting logic needed.
This is a direct, structural match between what these examples actually
are and the training pipeline's existing emphasis mechanism for exactly
the capability (post-interruption yield behavior) this whole effort
exists to improve.

`persona_manifest.py`'s `blueprint.utterances` field is validated for
shape (2-12 items, alternating speakers, starting user/ending assistant,
`Utterance` non-blank, `MAX_ASSISTANT_WORDS=45` per assistant turn) but
never read for its *content* by `persona_manifest.py`'s own logic --
only `.character`/`.split`/`.curriculum`/`.timing`/`.id` matter for
filtering, weighting, and the lookup key. A minimal, honest 2-utterance
blueprint is therefore sufficient and correct:

```
Utterance(speaker="user", text=probe.prompt)
Utterance(speaker="assistant", text=rollout.transcript)
```

(`rollout.transcript` truncated to `MAX_ASSISTANT_WORDS=45` words if it
exceeds that -- see Testing section for what to do if this ever
triggers, since it would mean losing real content the corpus otherwise
captured faithfully.)

## Curriculum mapping

Confirmed with the project owner (2026-07-31): map from the original
probe's own `category` field in `evals/persona/probes.yaml`, not a flat
default -- this preserves semantic meaning and lets lore/uncertainty
self-distillation examples *also* receive `persona_manifest.py`'s
existing curriculum-based reweighting (`TARGET_CURRICULA`), not just the
interruption-timing one.

```
lore              -> CurriculumKind.LORE           (exact match)
uncertainty       -> CurriculumKind.UNCERTAINTY     (exact match)
break_character   -> CurriculumKind.RESISTANCE      (both are about the
modern_redirect   -> CurriculumKind.RESISTANCE       character resisting
                                                       pressure to break
                                                       persona -- a real
                                                       semantic fit, not
                                                       a forced one)
concise           -> CurriculumKind.GENERAL         (no clean match --
                                                       brevity is a
                                                       different axis
                                                       than content
                                                       category, so it
                                                       correctly does
                                                       not receive a
                                                       reweight it
                                                       doesn't earn)
```

Any `probes.yaml` category not in this table (none exist today, but the
mapping function should not silently default an unrecognized category
to `GENERAL` -- raise, so a future new probe category doesn't get
mapped wrong) needs an explicit `ValueError` at build time.

## Provenance fields

`DialogueBlueprint.provenance: Provenance` requires
`teacher_model`/`teacher_revision`/`character_card_sha256`/
`prompt_template_revision`/`seed`/`voice_license`, all non-blank
strings (except `seed`, an int) -- these fields were designed around
the original TTS-generation pipeline's "teacher LLM produces dialogue,
`say` synthesizes audio" provenance model, which doesn't literally apply
to self-distillation (there is no teacher LLM or TTS voice; the "teacher"
is the live PersonaPlex model's own output, selected via reward
filtering). Chosen values, honest about what's actually true:

- `teacher_model`: `"personaplex-7b-v1-self-distillation"` -- names the
  real source (the live model itself), distinct from any TTS-pipeline
  teacher model string, so provenance is traceable and never confusable
  with the original corpus's meaning of this field.
- `teacher_revision`: the PersonaPlex image tag already recorded
  elsewhere in this project's configs (`configs/runtime.yaml`'s
  `personaplex_image`, e.g. `tolkien-voice/personaplex:3428dfd`) --
  reuse that value directly rather than inventing a new one, so it's
  traceable to the actual model version that generated this audio.
- `character_card_sha256`: `hashlib.sha256(card_path.read_bytes()).hexdigest()`
  over `configs/characters/{character}.yaml` -- the exact same
  computation `generate.py` already uses for the original corpus
  (`generate.py:319`), so this field means the same thing across both
  corpora.
- `prompt_template_revision`: `"self-distillation-v1"` -- there is no
  TTS prompt template involved; this labels the *scheme* (this specific
  blueprint-synthesis design), so a future revision to how these
  blueprints are built can be distinguished from this one.
- `seed`: the same `--seed` value `build-self-distillation-corpus.py`
  was invoked with for that rollout (already available; `collect_rollout_group`
  is seeded per-call from the script's `--seed` plus the rollout index).
- `voice_license`: `"self-distillation-v6-continuation"` -- reuses the
  exact literal already established for `RenderedExample.voice_license`
  in `self_distillation.py`, so both records agree.

## Blueprint `id`, `scenario`, and `canonical_hash`

`id` must exactly equal the corresponding `RenderedExample.id`
(`f"{scenario_id}-self-distill"`) -- `persona_manifest.py`'s
`_rows_by_id` matches on the audio filename's stem, so any mismatch
here silently drops the example from the built manifest with no error
(a scenario missing from the final weighted output, not a crash) --
this needs an explicit test, not just a code-review-time assertion.

`scenario` (required, non-blank, free text): the probe's own `prompt`
text -- already a meaningful, human-readable description of what this
example is about, and avoids inventing new prose.

`canonical_hash` is a `DialogueBlueprint` property (not a stored field)
computed from `utterances` -- automatically consistent as long as
`utterances` is set correctly; no separate handling needed.

## New derived dataset directory -- do not mutate the existing v6 corpus in place

The existing `source_dataset` (`$data_root/datasets/mvp-v6-decoupled`
on Spark) is the corpus every prior successful v6 training run already
used. Appending to it in place would be a real, consequential,
hard-to-cleanly-reverse mutation of shared, already-relied-upon data.

Instead: build a new, separate derived dataset directory (e.g.
`$data_root/datasets/mvp-v6-decoupled-self-distill-v1`) containing:

- `dialogues.jsonl`: the *union* of the original v6 corpus's
  `dialogues.jsonl` (copied verbatim, unchanged) plus the newly
  synthesized self-distillation blueprints, appended.
- `manifests/train.jsonl`: the union of the original v6 corpus's
  `manifests/train.jsonl` (copied verbatim) plus the self-distillation
  corpus's own `manifests/train.jsonl` rows, appended, with `path`
  values rewritten to point at where this new directory's own copy of
  the self-distillation audio lives (see below).
- `manifests/validation.jsonl`: copied verbatim from the original v6
  corpus, **completely unchanged** -- self-distillation examples are
  never added to validation. Evaluation must stay on data the model
  never trained on, self-distillation or not; the whole point of a
  held-out set is undermined by adding to it from the same process
  being evaluated.
- `audio/`: the original v6 corpus's audio directory's contents, plus
  the self-distillation corpus's WAV+JSON pairs *copied* in (not
  referenced across directories) -- self-contained, matches how the v6
  corpus's own audio is already organized, and avoids the new dataset
  directory silently depending on `/tmp/self-distill-*`'s scratch
  output surviving.

This is purely additive to a new directory -- the original
`mvp-v6-decoupled` corpus is never modified, so every prior run's
provenance and reproducibility is untouched, and this integration can
be discarded/redone without any risk to existing data.

## Scope

**In scope:** a new script/module that (1) synthesizes one
`DialogueBlueprint` per self-distillation `RenderedExample` per the
mapping above, (2) builds the new derived dataset directory (blueprint
union, train-manifest union with path rewriting, validation copied
unchanged, audio copied), (3) validates the result via this project's
existing `audit`/`conditioning` checks before declaring success (same
checks `train-mvp-v6-persona.sh` already runs, run here first so a
malformed derived corpus is caught before a training run is even
attempted).

**Out of scope:** actually invoking `train-mvp-v6-persona.sh` against
the new derived dataset, or any change to `persona_manifest.py`,
`audit.py`, `conditioning.py`, or the training script itself -- this
work only prepares a valid input; running training against it is a
separate, explicit decision.

## Testing and validation

- Unit tests for the curriculum-mapping function: each of the five known
  categories maps correctly; an unrecognized category raises.
- A test proving blueprint `id` exactly matches its `RenderedExample`'s
  `id` for a representative self-distillation manifest.
- A test proving the derived dataset's `dialogues.jsonl` validates
  against `DialogueBlueprint`'s own pydantic schema (every synthesized
  blueprint is a genuinely valid instance, not just structurally
  plausible).
- A test proving `validation.jsonl` is byte-identical to the source v6
  corpus's own `validation.jsonl` after the derive step -- the
  strongest possible guarantee that validation was never touched.
- Run this project's existing `tolkien_voice.training.audit` (with
  `--minimum-per-character 64`, matching `train-mvp-v6-persona.sh`'s
  own invocation) and `tolkien_voice.training.conditioning` against the
  derived dataset as the actual go/no-go check -- these are the same
  checks the training script itself runs, so passing them here is a
  direct, meaningful signal the derived corpus is trainable, not just
  schema-valid.
- If `rollout.transcript` ever exceeds `MAX_ASSISTANT_WORDS=45` and
  needs truncation for blueprint validity: log which scenario IDs this
  happened for (a real, visible signal that a small number of
  self-distillation examples lost content for blueprint-validity
  reasons, not a silent truncation) -- this should be rare (transcripts
  are yield-gated to fast responses) but must not be silent if it
  occurs.

## Reproducibility entry point

Run the new integration script/module against the self-distillation
corpus's output directory and the existing v6 `mvp-v6-decoupled`
dataset, producing the new derived dataset directory -- entirely on
Spark (or any machine with both corpora available), no GPU or live
PersonaPlex instance required. Once built, this derived dataset has the
exact same shape (`dialogues.jsonl` + `manifests/{train,validation}.jsonl`
+ `audio/`) as `train-mvp-v6-persona.sh`'s hardcoded `source_dataset`
path -- `train-mvp-v6-persona.sh` itself has no parameter for pointing
at a different corpus today, so actually running training against this
derived dataset needs either a small, separate change to that script
(a new argument or environment override) or manually substituting the
directory at that path -- deliberately left as a distinct, explicit next
decision outside this document's scope, not something this integration
work resolves on its own.
