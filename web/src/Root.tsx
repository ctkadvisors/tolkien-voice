import { useCallback, useEffect, useState } from "react";

import { endSession, getCharacters, getSession, startSession } from "./api";
import { CharacterSelect } from "./pages/CharacterSelect";
import { ConversationPage } from "./pages/ConversationPage";
import type { Character, Session } from "./types";


const IDLE_SESSION: Session = { state: "idle" };


export function App() {
  const [characters, setCharacters] = useState<Character[]>([]);
  const [session, setSession] = useState<Session>(IDLE_SESSION);
  const [selectedCharacterId, setSelectedCharacterId] = useState("");
  const [selectedVoiceId, setSelectedVoiceId] = useState("");
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    let active = true;
    void Promise.all([getCharacters(), getSession()])
      .then(([loadedCharacters, loadedSession]) => {
        if (!active) return;
        setCharacters(loadedCharacters);
        setSession(loadedSession);
        const initial = loadedCharacters[0];
        if (initial) {
          setSelectedCharacterId(initial.id);
          setSelectedVoiceId(initial.default_voice);
        }
      })
      .catch((cause: unknown) => {
        if (active) setError(cause instanceof Error ? cause.message : "Unable to open the listening room");
      });
    return () => {
      active = false;
    };
  }, []);

  const awaitingModel = session.state === "waiting_for_lease" || session.state === "loading";
  useEffect(() => {
    if (!awaitingModel) return;
    const timer = window.setInterval(() => {
      void getSession().then(setSession).catch(() => undefined);
    }, 1000);
    return () => window.clearInterval(timer);
  }, [awaitingModel]);

  const handleStart = useCallback(async () => {
    if (!selectedCharacterId || !selectedVoiceId) return;
    setError(null);
    try {
      setSession(await startSession(selectedCharacterId, selectedVoiceId));
    } catch (cause) {
      setError(cause instanceof Error ? cause.message : "Conversation could not start");
    }
  }, [selectedCharacterId, selectedVoiceId]);

  const handleEnd = useCallback(async () => {
    if (!session.session_id) return;
    try {
      await endSession(session.session_id);
    } finally {
      setSession(IDLE_SESSION);
    }
  }, [session.session_id]);

  if (session.state !== "idle" && session.session_id) {
    return <ConversationPage session={session} onEnd={handleEnd} />;
  }

  // A failed activation bounces back to idle with the failure recorded on the
  // session payload (session.py's _last_failure) -- surface it instead of
  // silently resetting to the default blurb.
  const sessionFailure = session.error
    ? session.blocking_owner
      ? `${session.error} (in use by ${session.blocking_owner.name})`
      : session.error
    : null;

  return (
    <CharacterSelect
      characters={characters}
      selectedCharacterId={selectedCharacterId}
      selectedVoiceId={selectedVoiceId}
      onSelectionChange={(characterId, voiceId) => {
        setSelectedCharacterId(characterId);
        setSelectedVoiceId(voiceId);
      }}
      onVoiceChange={setSelectedVoiceId}
      onStart={() => void handleStart()}
      disabled={!characters.length}
      statusMessage={error ?? sessionFailure ?? undefined}
    />
  );
}

