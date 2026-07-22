# Hugging Face access for PersonaPlex

PersonaPlex model weights are gated separately from the MIT-licensed source. Deployment cannot download or start the model until the repository owner completes this one-time prerequisite.

1. Sign in to Hugging Face and accept the NVIDIA Open Model License on [nvidia/personaplex-7b-v1](https://huggingface.co/nvidia/personaplex-7b-v1).
2. Create a fine-grained token that has read access only to `nvidia/personaplex-7b-v1`.
3. On the Spark, create `/home/chrisknuteson/tolkien-voice-data/secrets` with mode `0700`.
4. Write the token to `/home/chrisknuteson/tolkien-voice-data/secrets/hf_token` without placing it in shell history, then set the file mode to `0600`.

An interactive safe installation method is:

```bash
install -d -m 0700 /home/chrisknuteson/tolkien-voice-data/secrets
read -rsp 'Hugging Face token: ' token_value
printf '\n'
install -m 0600 /dev/null /home/chrisknuteson/tolkien-voice-data/secrets/hf_token
printf '%s\n' "$token_value" > /home/chrisknuteson/tolkien-voice-data/secrets/hf_token
unset token_value
```

The deployment preflight reports only whether access succeeds. It must never echo the token, pass it as a command-line argument, store it in Git, include it in Docker image metadata, or return it to the browser.

Verify file presence and permissions without printing its contents:

```bash
test -s /home/chrisknuteson/tolkien-voice-data/secrets/hf_token
test "$(stat -c '%a' /home/chrisknuteson/tolkien-voice-data/secrets/hf_token)" = 600
printf 'PersonaPlex credential file is installed with mode 0600.\n'
```
