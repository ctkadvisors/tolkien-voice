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

