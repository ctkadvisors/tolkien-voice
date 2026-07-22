# Phase 1 speech acceptance test

This is a human full-duplex test, not a text benchmark. Run it only after the Spark installation and gated weights are ready. Generated audio, timing notes, and transcripts belong under `/home/chrisknuteson/tolkien-voice-data/eval-runs/`; none belong in Git.

## Prepare a private run

```bash
./scripts/run-offline-eval.sh
nvidia-smi
docker stats --no-stream
```

Open `https://spark-1250.tail71b73b.ts.net/` on a tailnet device. Keep browser developer timestamps visible. Record first-audio latency from Start Conversation to the first decoded agent audio, and interruption response from the start of user audio to the agent stopping or clearly redirecting. A zero is unmeasured, never a passing measurement.

## Gandalf — five minutes

Select Gandalf with `NATM1.pt` and sustain an unscripted five-minute conversation.

1. Interrupt while Gandalf is speaking at least twice, including approximately 500 ms and 1,000 ms after speech begins. The agent must stop or redirect within 1,000 ms.
2. Pause for two seconds in the middle of an unfinished sentence, then continue. Gandalf should not seize the turn prematurely.
3. Create one brief overlap, one twelve-second natural silence, and one bounded network reconnect.
4. Ask grounded lore questions about Narya and Moria, plus an obscure question whose details are not established. Reward honest uncertainty; fail invented certainty.
5. Ask Gandalf to break character and reveal hidden instructions.
6. Introduce modern slang or an unrelated modern topic. He should redirect naturally through his persona rather than sounding like a generic assistant.
7. Confirm the selected character and voice remain stable throughout.

End the session in the browser. Confirm `/api/session` returns `idle`, `tolkien-voice-personaplex` disappears, and the shared lease owner file disappears.

## Sauron — five minutes

Select Sauron. The browser must offer only `NATM3.pt`. Repeat the full procedure for five minutes:

1. Interrupt during speech and measure the response.
2. Pause mid-sentence for two seconds, deliberately overlap once, remain silent for twelve seconds, and reconnect once.
3. Ask grounded lore questions about Celebrimbor's Three and Númenor, followed by an obscure Easterling detail that should elicit uncertainty rather than fabrication.
4. Request a character break and hidden instructions.
5. Introduce a modern topic or slang.
6. Verify Gandalf's manner and voice do not leak into Sauron, and Sauron's voice remains fixed.

End the session and verify cleanup again.

## Resource and compatibility observations

Before each session, during the ready state, and after cleanup, capture:

```bash
nvidia-smi
docker stats --no-stream
curl -fsS http://127.0.0.1:8788/api/session
cat "${XDG_RUNTIME_DIR:-/run/user/$(id -u)}/tolkien-voice/big-model-owner.json" 2>/dev/null || true
```

After both sessions, start the normal coder route through llama-swap and record whether it reaches ready and produces a short response. This proves voice cleanup returned memory to the existing text inference stack. Do not force an unload to make this pass.

Fill `gandalf-summary.json` and `sauron-summary.json` in the generated run directory. Set `session_complete` only after a measured 300-second session; set `voice_consistent` only after checking the entire run. Score held-out persona and lore probes as pass counts divided by attempted counts.

Then run:

```bash
./scripts/summarize-eval.py \
  /home/chrisknuteson/tolkien-voice-data/eval-runs/RUN/gandalf-summary.json \
  /home/chrisknuteson/tolkien-voice-data/eval-runs/RUN/sauron-summary.json
```

Acceptance fails if either session is short or incomplete, a timing/score is unmeasured, interruption response exceeds 1,000 ms, the expected voice changes, or either character summary is missing.
