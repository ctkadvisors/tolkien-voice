# Vendored upstreams

`vendor/personaplex` is a Git submodule of NVIDIA's official PersonaPlex source:

- Repository: <https://github.com/NVIDIA/personaplex>
- Pinned commit: `3428dfd95309a7f3c84fd93259ded0f810d1ff91`
- Source license: MIT; retain the upstream license and copyright notices.
- Model: `nvidia/personaplex-7b-v1`, distributed separately under the gated NVIDIA Open Model License.

Update only through an explicit reviewed change:

```bash
git -C vendor/personaplex fetch origin
git -C vendor/personaplex checkout COMMIT_SHA_REVIEWED_FOR_THE_UPDATE
git add vendor/personaplex
```

Never commit model weights, Hugging Face credentials, generated audio, or voice samples.

`vendor/moshi-finetune` is a Git submodule of Kyutai's official Moshi LoRA
training toolkit:

- Repository: <https://github.com/kyutai-labs/moshi-finetune>
- Pinned commit: `2acc879fe7c48f885a18f6cc9548bccb2674d87b`
- Source license: Apache License 2.0.

PersonaPlex's published configuration is not directly consumable by this
trainer. `configs/training/personaplex-7b-v1.json` reconstructs the exact
architecture from the pinned NVIDIA loader, and the metadata compatibility
probe must pass before a training process is started.
