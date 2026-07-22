export type SessionState =
  | "idle"
  | "waiting_for_lease"
  | "loading"
  | "ready"
  | "connected"
  | "disconnected"
  | "stopping"
  | "failed";

export type Character = {
  id: string;
  display_name: string;
  summary: string;
  default_voice: string;
  allowed_voices: string[];
};

export type Session = {
  state: SessionState;
  session_id?: string;
  character_id?: string;
  character_name?: string;
  voice_id?: string;
  blocking_owner?: {
    kind: string;
    name: string;
  };
  error?: string;
};

