# Tolkien STS training smoke results

Date: 2026-07-22  
Hardware: NVIDIA DGX Spark / GB10  
Status: training and native-adapter feasibility passed; quality MVP not yet passed

## What ran

- Teacher: local Tolkien 27B v4 Q8 through the OpenAI-compatible `tolkien` route.
- Characters: Aragorn and Sauron.
- Private smoke corpus: two four-turn dialogues rendered as 24 kHz stereo PCM16.
- Backbone: `nvidia/personaplex-7b-v1` at Hugging Face revision `fdaf4090a61cb315c138a1faee287ffd6c716309`.
- PersonaPlex source: `3428dfd95309a7f3c84fd93259ded0f810d1ff91`.
- Moshi source: `061cc4c630d9e11722e08b7d02b1836ba58f30e8`.
- Moshi-Finetune source: `2acc879fe7c48f885a18f6cc9548bccb2674d87b`.
- Training: BF16, LoRA rank 8, scaling 2.0, batch size 1, 12-second crop, gradient checkpointing, one optimizer step.

The private audio, checkpoint, metrics, and run metadata remain outside Git under `/home/chrisknuteson/tolkien-voice-data` on Spark.

## Measured training result

| Measurement | Result |
|---|---:|
| Optimizer steps | 1 |
| Loss | 3.4820823669 |
| Peak allocated GPU memory | 16.814 GB |
| Allocated GPU memory after step | 16.465 GB |
| Trainer throughput | 265.608 words/s |
| Total trainer wall time | 1m 52s |
| Adapter size | 62,446,592 bytes |
| LoRA tensors | 1,090 |
| LoRA parameters | 31,156,224 |
| Adapter SHA-256 | `706f133ace4eb3fa6a6484147f850fc13eee078e7c942840d9635e193544eb52` |

The SHA was independently rechecked after training. `run-metadata.json` binds it to the exact base and source revisions and is mode `0600`.

## Native inference proof

NVIDIA's pinned PersonaPlex fork stores attention projections in a legacy fused form, while the pinned trainer emits current Moshi split-projection LoRA tensors. The runtime bridge reconstructs each fused delta, checks every target and shape, and refuses partial loads.

The rebuilt runtime reported:

```json
{
  "status": "ok",
  "adapter_sha256": "706f133ace4eb3fa6a6484147f850fc13eee078e7c942840d9635e193544eb52",
  "base_revision": "fdaf4090a61cb315c138a1faee287ffd6c716309",
  "merged_projection_pairs": 545
}
```

An adapter-backed Aragorn session then completed the native PersonaPlex binary handshake, accepted Opus made from the held-out user channel, and returned two native audio messages totaling 100 encoded bytes. This path is browser/client audio → PersonaPlex/Moshi → audio; it does not contain ASR, a text LLM turn, or TTS.

## What this does not prove

This one-step, two-dialogue adapter is a plumbing and trainability proof. It does not establish that Tolkien behavior improved, that Aragorn and Sauron remain distinct, or that a casual human conversation is responsive and clean. It is not the release candidate and must not be presented as one.

The accepted MVP still requires:

1. A larger balanced private corpus and a short real training run.
2. Blinded base-vs-adapter held-out persona and lore evaluation.
3. Character-confusion, first-audio, underrun, and interruption gates.
4. Browser microphone meter/device selection, bounded playout buffering, barge-in, and five-minute human sessions for Aragorn and Sauron.
