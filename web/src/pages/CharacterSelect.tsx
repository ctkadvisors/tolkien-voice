import type { Character } from "../types";


type Props = {
  characters: Character[];
  selectedCharacterId: string;
  selectedVoiceId: string;
  onSelectionChange: (characterId: string, defaultVoiceId: string) => void;
  onVoiceChange: (voiceId: string) => void;
  onStart: () => void;
  disabled?: boolean;
  statusMessage?: string;
};


const marks: Record<string, string> = {
  gandalf: "✦",
  sauron: "◉",
  galadriel: "☽",
  samwise: "❧",
  aragorn: "♔",
  frodo: "○",
  treebeard: "♧",
  boromir: "◇",
  barliman: "⌂",
};


function voiceLabel(voice: string): string {
  return voice.replace(".pt", "").replace("NATM", "Voice M").replace("NATF", "Voice F");
}


export function CharacterSelect({
  characters,
  selectedCharacterId,
  selectedVoiceId,
  onSelectionChange,
  onVoiceChange,
  onStart,
  disabled = false,
  statusMessage,
}: Props) {
  const selected =
    characters.find((character) => character.id === selectedCharacterId) ?? characters[0];

  return (
    <main className="threshold-shell">
      <header className="threshold-header reveal-one">
        <p className="eyebrow">A private audience on the Spark</p>
        <h1>Whom would you speak with?</h1>
        <p className="lede">
          Choose one voice from Middle-earth. They listen while they speak—and yield when you do.
        </p>
      </header>

      <section className="character-grid reveal-two" aria-label="Characters">
        {characters.map((character, index) => {
          const active = character.id === selected?.id;
          return (
            <button
              className={`character-card ${active ? "is-selected" : ""}`}
              key={character.id}
              type="button"
              aria-pressed={active}
              aria-label={`Select ${character.display_name}`}
              onClick={() => onSelectionChange(character.id, character.default_voice)}
              style={{ "--order": index } as React.CSSProperties}
            >
              <span className="character-mark" aria-hidden="true">{marks[character.id] ?? "✧"}</span>
              <span className="character-copy">
                <strong>{character.display_name}</strong>
                <small>{character.summary}</small>
              </span>
              <span className="selection-rune" aria-hidden="true">{active ? "◆" : "◇"}</span>
            </button>
          );
        })}
      </section>

      {selected ? (
        <section className="summoning-bar reveal-three" aria-label="Conversation settings">
          <div>
            <span className="field-kicker">Chosen voice</span>
            <strong>{selected.display_name}</strong>
          </div>
          <label className="voice-field">
            <span>Voice</span>
            <select value={selectedVoiceId} onChange={(event) => onVoiceChange(event.target.value)}>
              {selected.allowed_voices.map((voice) => (
                <option key={voice} value={voice}>{voiceLabel(voice)}</option>
              ))}
            </select>
          </label>
          <button className="summon-button" type="button" onClick={onStart} disabled={disabled}>
            <span>Start conversation</span>
            <span aria-hidden="true">→</span>
          </button>
        </section>
      ) : null}

      <p className="privacy-note" role="status" aria-live="polite">
        {statusMessage ?? "Audio stays on your private tailnet and is not recorded."}
      </p>
    </main>
  );
}

