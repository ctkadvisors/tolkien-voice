# Tolkien-Native STS Distillation MVP

Date: 2026-07-22

Status: Approved by prior user direction; corrective replacement for the
prompt-only MVP

## Outcome

The first accepted MVP is a native, full-duplex speech-to-speech checkpoint
adapted from the behavior of the existing Tolkien 27B v4 model. A role prompt
on an unchanged PersonaPlex checkpoint is only the control arm and does not
qualify as the MVP.

The live response path remains native STS:

```text
microphone -> Tolkien STS checkpoint + adapter -> speaker
```

The Tolkien 27B model participates offline as teacher and judge. It does not
sit in the live audio path.

## First proof

The smallest credible proof must produce all of the following:

1. A versioned, provenance-bearing set of short Tolkien-teacher conversations.
2. Corresponding 24 kHz stereo audio where the left channel is the model and
   the right channel is the user, including pauses, overlap and interruption.
3. Timestamped assistant transcripts in the format consumed by the official
   Moshi fine-tuning pipeline.
4. A LoRA checkpoint trained from the pinned PersonaPlex 7B v1 weights on the
   DGX Spark.
5. Native inference with that adapter loaded.
6. A blind A/B report comparing the unchanged checkpoint with the adapter on
   held-out persona, lore, concision and duplex probes.

The initial training corpus is intentionally small. Aragorn and Sauron are the
required characters because they provide a useful contrast in manner. The
pipeline and manifest support the full existing roster from the start.

## Training architecture

```text
Tolkien 27B v4 teacher on Spark
        |
        | structured short dialogues conditioned by character cards
        v
Dialogue blueprints + held-out split
        |
        | licensed/local synthetic voices + timing renderer
        v
24 kHz stereo conversations + word alignments
        |
        | official Moshi LoRA fine-tuning, Mimi frozen
        v
PersonaPlex 7B v1 + Tolkien STS LoRA
        |
        +--> native full-duplex browser inference
        +--> A/B evaluation against unchanged PersonaPlex
```

The official `kyutai-labs/moshi-finetune` repository is pinned as a separate
upstream. PersonaPlex's published `config.json` is only an identifier, while
the trainer expects the full LM architecture. The repository therefore owns
an explicit PersonaPlex training config derived from the pinned NVIDIA loader.
A compatibility probe must compare checkpoint keys and shapes before any
training step. The adapter is never claimed compatible merely because the
base models share Moshi ancestry.

## Data contract

Every dialogue blueprint records:

- stable ID and schema version;
- character-card hash and Tolkien teacher model/revision;
- generation seed and prompt-template revision;
- ordered user/assistant utterances;
- timing intent: normal turn, pause, overlap or interruption;
- train, validation or held-out split;
- source and voice-license metadata.

Rendered examples contain a 24 kHz stereo WAV, adjacent alignment JSON and a
Moshi JSONL entry. Generated data, model weights and raw evaluation recordings
remain outside Git under `tolkien-voice-data`.

The held-out probes are authored before generation and their normalized text
hashes are rejected from the training split. The training manifest fails
closed on duplicate examples, missing provenance, invalid speaker order,
unlicensed voices, clipping, wrong channel count or wrong sample rate.

## Multi-persona behavior

The first experiment uses a shared LoRA trained on balanced Aragorn and Sauron
examples while retaining PersonaPlex role prompts at inference. This tests
whether role prompting preserves character separation after the shared Tolkien
adaptation. Character-confusion rate is an explicit gate.

If the shared adapter fails that gate, the already-rendered manifest is split
into per-character LoRAs. The product still supports multiple personas by
selecting an adapter at session start; no text LLM is introduced into the live
path.

## Acceptance gates

The result is an MVP only when:

- at least one non-empty LoRA checkpoint is produced by an optimizer step on
  the PersonaPlex checkpoint;
- the native server loads that adapter and completes a binary audio round trip;
- held-out persona/lore scoring improves over the unchanged base, with the
  exact sample count and confidence limits reported;
- neither character is identified as the other in more than 5% of held-out
  responses;
- speech remains full duplex and a user interruption suppresses or redirects
  agent speech within 1,000 ms in the test harness;
- first-audio latency and audio underruns do not regress by more than 10% from
  the unchanged checkpoint;
- a human can select an input device, see input level, interrupt the agent and
  end the conversation without the agent continuing to speak.

Until these gates pass, the repository must label the work as a training
experiment, not a working Tolkien STS MVP.

## Resource contract

Dataset generation, training and inference each acquire the existing Spark
large-model lease. Tolkien teacher generation and PersonaPlex training cannot
run concurrently. Training outputs are written atomically beneath
`/home/chrisknuteson/tolkien-voice-data/training-runs`; failed runs preserve
logs and never replace the active adapter.

The first Spark feasibility run uses a tiny dataset and one to five optimizer
steps. Its purpose is to prove checkpoint compatibility, backward pass,
checkpoint save and adapter reload before authorizing a longer run.

## Non-goals for the first proof

- No live Tolkien 27B inference stage.
- No iPhone deployment.
- No voice cloning from actors or audiobook narrators.
- No hundreds-of-hours corpus before the tiny proof beats the control.
- No claim that prompt-only PersonaPlex is the trained model.
