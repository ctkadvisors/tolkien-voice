import { useState } from "react";
import { render, screen } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { describe, expect, it, vi } from "vitest";

import { CharacterSelect } from "./CharacterSelect";
import type { Character } from "../types";


const characters: Character[] = [
  ["gandalf", "Gandalf", "NATM1.pt", ["NATM1.pt", "NATM2.pt"]],
  ["sauron", "Sauron", "NATM3.pt", ["NATM3.pt"]],
  ["galadriel", "Galadriel", "NATF2.pt", ["NATF2.pt", "NATF3.pt"]],
  ["samwise", "Samwise Gamgee", "NATM0.pt", ["NATM0.pt", "NATM1.pt"]],
  ["aragorn", "Aragorn", "NATM2.pt", ["NATM2.pt", "NATM1.pt"]],
  ["frodo", "Frodo Baggins", "NATM0.pt", ["NATM0.pt", "NATM2.pt"]],
  ["treebeard", "Treebeard", "NATM3.pt", ["NATM3.pt", "NATM1.pt"]],
  ["boromir", "Boromir", "NATM2.pt", ["NATM2.pt", "NATM3.pt"]],
  ["barliman", "Barliman Butterbur", "NATM1.pt", ["NATM1.pt", "NATM0.pt"]],
].map(([id, display_name, default_voice, allowed_voices]) => ({
  id: id as string,
  display_name: display_name as string,
  summary: `A short description of ${display_name}.`,
  default_voice: default_voice as string,
  allowed_voices: allowed_voices as string[],
}));


function Harness({ onStart }: { onStart: (character: string, voice: string) => void }) {
  const [characterId, setCharacterId] = useState("gandalf");
  const [voiceId, setVoiceId] = useState("NATM1.pt");
  return (
    <CharacterSelect
      characters={characters}
      selectedCharacterId={characterId}
      selectedVoiceId={voiceId}
      onSelectionChange={(character, voice) => {
        setCharacterId(character);
        setVoiceId(voice);
      }}
      onVoiceChange={setVoiceId}
      onStart={() => onStart(characterId, voiceId)}
    />
  );
}


describe("CharacterSelect", () => {
  it("renders the complete initial roster", () => {
    render(<Harness onStart={vi.fn()} />);

    for (const character of characters) {
      expect(screen.getByRole("button", { name: new RegExp(character.display_name) })).toBeVisible();
    }
  });

  it("limits Sauron to the server-approved voice", async () => {
    const user = userEvent.setup();
    render(<Harness onStart={vi.fn()} />);

    await user.click(screen.getByRole("button", { name: /Sauron/ }));

    const voice = screen.getByLabelText("Voice") as HTMLSelectElement;
    expect(Array.from(voice.options).map((option) => option.value)).toEqual(["NATM3.pt"]);
  });

  it("starts with character and voice IDs only", async () => {
    const onStart = vi.fn();
    const user = userEvent.setup();
    render(<Harness onStart={onStart} />);

    await user.click(screen.getByRole("button", { name: /Sauron/ }));
    await user.click(screen.getByRole("button", { name: "Start conversation" }));

    expect(onStart).toHaveBeenCalledWith("sauron", "NATM3.pt");
    expect(JSON.stringify(onStart.mock.calls)).not.toContain("prompt");
  });
});

