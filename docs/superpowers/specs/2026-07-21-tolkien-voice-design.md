# Tolkien Voice: Native Full-Duplex Multi-Persona Speech

Date: 2026-07-21

Status: Proposed for implementation

Repository: `ctkadvisors/tolkien-voice` (private)

Deployment target: DGX Spark `spark-1250.tail71b73b.ts.net`

## Purpose

Build a native speech-to-speech Tolkien experience using PersonaPlex/Moshi so
that a person can hold a casual, low-latency, full-duplex conversation with a
selected Tolkien character. The system must listen while speaking, permit
natural interruption and overlap, and avoid an ASR-to-text-LLM-to-TTS runtime
pipeline.

The first release establishes a usable prompt-conditioned PersonaPlex
baseline on the Spark. The second release distills the existing Tolkien 27B
model's behavior into PersonaPlex with a multi-character LoRA adapter. The
baseline is required so training is driven by measured failures rather than
assumptions.

## Product decisions

- One shared speech checkpoint supports the full character roster.
- One character is active per live conversation.
- A new conversation may select a different character.
- In-session character switching is deferred until the single-character
  session is reliable; a future switch will reset speech state before loading
  the new role and voice prompts.
- Multiple characters speaking simultaneously is out of scope for the first
  release because PersonaPlex exposes one agent audio stream.
- The runtime path is native PersonaPlex speech-to-speech. The Tolkien 27B
  model is a training-data teacher and offline evaluator, not a runtime stage.
- The deployment is private over Tailscale and does not expose the speech
  service to the public internet.
- No actor, audiobook narrator, or other third-party voice will be cloned
  without explicit authorization. Initial characters use included PersonaPlex
  voices or original licensed voice samples.

## Initial character roster

The registry begins with:

- Gandalf
- Sauron
- Galadriel
- Samwise Gamgee
- Aragorn
- Frodo Baggins
- Treebeard
- Boromir
- Barliman Butterbur

Each character is represented by a versioned character card containing:

- Stable identifier and display name.
- Role, history, manner, vocabulary and behavioral boundaries.
- Lore scope and uncertainty behavior.
- Prompt-injection and break-character response guidance.
- Default PersonaPlex voice-prompt identifier.
- Optional path to an authorized custom voice sample.
- Evaluation categories and character-specific probes.

Character cards are data, not branching application code. Adding a character
must not require changes to the speech server.

## User experience

The browser page provides:

1. A character selector with name and short description.
2. A voice selector limited to voices allowed for that character.
3. A prominent Start Conversation action.
4. Microphone permission and connection state.
5. Live connection, listening, speaking and reconnecting indicators.
6. A transcript panel for diagnosis that can be hidden during conversation.
7. End Conversation, which releases the model and unified memory.

Starting a conversation reserves the Spark's large-model slot, launches or
warms PersonaPlex, applies the selected role and voice prompts, and connects
the full-duplex audio stream. Ending the conversation tears down session state
and makes the large-model slot available to llama-swap.

The initial URL is private and uses the Spark's Tailscale MagicDNS identity.
The implementation should use valid Tailscale HTTPS where practical because
browser microphone access requires a secure context. A self-signed upstream
certificate may remain internal behind the Tailscale proxy.

## Architecture

```text
Browser microphone and speaker
            |
            | secure WebSocket/WebAudio over Tailscale
            v
Tolkien Voice session controller and character registry
            |
            | role prompt + voice prompt + duplex audio
            v
PersonaPlex 7B / Mimi codec on DGX Spark
            |
            +--> generated agent text for transcript/evaluation
            +--> generated 24 kHz agent audio for immediate playback
```

PersonaPlex receives and generates audio continuously. Its three model streams
are user audio, agent text and agent audio. The role prompt conditions the
agent text stream; the voice prompt conditions the agent audio stream. No
external ASR, Tolkien LLM or TTS service participates in live response
generation.

### Component boundaries

#### Character registry

Loads and validates character cards, builds the PersonaPlex text prompt and
resolves an allowed voice prompt. It contains no model-serving logic.

#### Session controller

Owns the lifecycle of one active voice session. It validates character and
voice choices, acquires the Spark large-model lease, starts PersonaPlex,
returns connection metadata to the browser and releases resources on explicit
end, disconnect timeout or process failure.

#### PersonaPlex runtime

Pins the upstream NVIDIA PersonaPlex source and model revision. It runs in a
GPU-enabled container with persistent Hugging Face and compilation caches.
The service listens only on the host or Tailscale-facing path selected by the
session controller.

#### Browser client

Provides character selection, microphone capture, audio playback and state
indicators. It consumes the existing PersonaPlex WebSocket protocol rather
than creating a second speech protocol.

#### Evaluation harness

Runs recorded duplex probes and scores both the generated transcript and
conversation dynamics. Evaluation is separate from the interactive service.

## Spark resource and lifecycle contract

The Spark has 128 GB unified memory and also serves an 80B-class Qwen coder.
PersonaPlex and the coder must be treated as mutually exclusive large-model
workloads even if both happen to load in a lightly used state.

The controller must:

- Coordinate with the existing llama-swap `big` model group or an equivalent
  single-owner lease before starting PersonaPlex.
- Stop any resident large text model before loading PersonaPlex.
- Prevent a large text model from starting while an active voice conversation
  owns the lease.
- Keep the embedding service resident.
- Stop PersonaPlex on explicit End Conversation.
- Stop PersonaPlex after a configurable disconnected-idle timeout, initially
  five minutes.
- Never kill an active speech conversation merely because a coder request
  arrives; the coder request must wait or return a clear busy response.
- Recover the lease after a crashed or abandoned session.

No model is deleted during switching. Persistent model and compilation caches
remain on NVMe.

## Authentication and model access

`nvidia/personaplex-7b-v1` is gated by the NVIDIA Open Model License. Before
deployment, the repository owner must accept the terms on Hugging Face and
provide a fine-grained read token. The token is installed as a Spark secret or
credential file and is never committed, logged, returned to the browser or
baked into a container image.

The service itself is reachable only through the authenticated Tailscale
network. The initial release does not add application-level accounts.

## Repository layout

```text
configs/
  characters/                 # versioned role and voice cards
  runtime.yaml                # ports, idle timeout, model revision
docs/
  operations/                 # Spark deployment and rollback runbooks
  superpowers/specs/          # approved designs
evals/
  duplex/                     # interruption, pause and overlap probes
  persona/                    # transcript-level persona and lore probes
scripts/
  deploy-spark.sh
  start-session.sh
  stop-session.sh
  smoke-test.sh
src/tolkien_voice/
  characters.py
  controller.py
  lease.py
  settings.py
web/                          # character-aware PersonaPlex browser client
container/                    # pinned runtime image definition
tests/
```

Model weights, voice recordings, generated datasets, Hugging Face tokens,
evaluation recordings and other large or sensitive artifacts are excluded
from Git and stored on the Spark under a configurable data root.

## Delivery phases

### Phase 1: prompt-conditioned conversational baseline

- Pin PersonaPlex source and model revisions.
- Build a GB10-compatible runtime image.
- Add the character registry, including Sauron.
- Add private HTTPS browser access and character selection.
- Implement the Spark large-model lease and idle cleanup.
- Verify microphone, speaker, interruption and reconnect behavior.
- Record prompt-only persona and duplex baseline scores.

This phase produces the first usable live speech conversation. It is honest
about the model: PersonaPlex is prompted with Tolkien character cards but has
not yet absorbed Tolkien 27B behavior through training.

### Phase 2: Tolkien behavior distillation

- Extend `persona-forge` exports into spoken-dialogue manifests.
- Use Tolkien 27B v4 to generate multi-turn character conversations,
  grounded lore answers, provocations, corrections and rejected responses.
- Generate authorized duplex audio with varied user voices, timing, pauses,
  overlaps and interruptions.
- Store 24 kHz stereo audio with timestamped transcripts.
- Fine-tune from PersonaPlex with Moshi LoRA while preserving the Mimi codec,
  hybrid prompts and general conversational replay data.
- Gate the adapter against prompt-only PersonaPlex and Tolkien 27B transcript
  baselines.

The pilot target is 20-50 hours of audio. Dataset size expands only when the
pilot demonstrates improved persona behavior without degrading duplex timing.

### Phase 3: hardening

- Expand training based on measured failure categories.
- Add authorized custom voice prompts.
- Add longer-session and recovery tests.
- Add observability for startup, real-time factor, memory and session errors.
- Produce a reproducible adapter release and rollback procedure.

On-device iPhone deployment and multiple simultaneous characters are separate
future designs and are not required for these phases.

## Data generation requirements

Each distilled example must preserve:

- Character card and role prompt.
- Authorized voice-prompt reference.
- User and agent audio channels.
- Transcript text and timestamps.
- Turn boundaries, overlap and interruption annotations.
- Lore context used by the Tolkien teacher.
- Dataset provenance and license metadata.
- Train, validation or held-out evaluation assignment.

Held-out persona prompts, lore questions and duplex timing scenarios must not
appear in training generation inputs.

General PersonaPlex/Moshi conversation replay is mixed into training to reduce
catastrophic forgetting of pause handling, listening while speaking and
natural turn-taking.

## Error handling

- Missing or unauthorized Hugging Face credentials: refuse startup with a
  clear operator action; never retry indefinitely.
- Large-model lease busy: show which workload owns the Spark and allow the
  user to retry; never evict an active conversation.
- Model cold start: show loading progress and keep the browser connection
  alive within a bounded startup timeout.
- Microphone denied: retain the page and provide browser permission guidance.
- WebSocket interruption: attempt bounded reconnect to the same session;
  release the lease after the disconnected-idle timeout.
- Model process failure: capture logs, release the lease and present a clean
  restart action.
- Invalid character or voice identifier: reject before starting the model.

## Privacy and safety

- Audio and transcripts stay on the user's Tailscale network.
- Interactive audio is not persisted by default.
- Evaluation recording requires an explicit recording mode and visible UI
  indication.
- Secrets and model credentials never enter Git.
- Character cards distinguish fictional roleplay from claims of real identity.
- Voice assets require documented authorization and provenance.

## Testing strategy

### Automated tests

- Character card schema and prompt construction.
- Unknown character and unauthorized voice rejection.
- Lease acquisition, contention, crash recovery and idle release.
- Runtime configuration and secret redaction.
- Session start/stop state transitions.
- Container health and WebSocket handshake.
- No persistent audio outside explicit recording mode.

### Network integration tests

- Start PersonaPlex from an idle Spark and confirm the selected role/voice.
- Confirm the embedding service remains healthy.
- Attempt a coder start during an active voice session and confirm it waits or
  returns busy without killing speech.
- End the voice session, confirm PersonaPlex exits and then confirm the coder
  can start normally.
- Verify private HTTPS and WebSocket access through Tailscale.

### Human acceptance test

For Gandalf and Sauron independently:

1. Hold a five-minute unscripted conversation.
2. Interrupt while the character is speaking and observe a natural stop or
   response.
3. Pause mid-sentence without causing premature takeover.
4. Ask several grounded lore questions, including obscure proper nouns.
5. Request that the model break character.
6. Introduce modern slang or an unrelated topic.
7. Verify the selected character and voice do not leak into the other.

## Success criteria

Phase 1 is accepted when:

- The private browser client can start and end a conversation without shell
  access.
- Gandalf and Sauron can each sustain a five-minute full-duplex session.
- User interruption stops or redirects agent speech within one second.
- The selected role and voice persist for the session.
- Ending or timing out a session releases PersonaPlex memory.
- Qwen coder on-demand behavior still works after voice-session cleanup.
- No audio is persisted by default.

Phase 2 is accepted when:

- The LoRA adapter improves the held-out persona/lore aggregate over
  prompt-only PersonaPlex.
- Character assignment and voice separation pass for every roster character.
- Duplex timing metrics do not regress materially from the Phase 1 baseline.
- The adapter, dataset manifest and evaluation report are reproducible from
  documented commands.

## Rollback

Deployment changes are additive. Rollback stops and disables the Tolkien Voice
service, removes its Tailscale proxy route and large-model registration, and
restores the previous llama-swap configuration from a timestamped backup.
Existing Qwen, embedding and Tolkien GGUF model files remain untouched.

