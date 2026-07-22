import type { Character, Session } from "./types";


async function request<T>(path: string, init?: RequestInit): Promise<T> {
  const response = await fetch(path, {
    ...init,
    cache: "no-store",
    headers: {
      "Content-Type": "application/json",
      ...init?.headers,
    },
  });
  if (!response.ok) {
    const body = await response.json().catch(() => ({}));
    const detail = typeof body.detail === "string" ? body.detail : `Request failed (${response.status})`;
    throw new Error(detail);
  }
  if (response.status === 204) {
    return undefined as T;
  }
  return response.json() as Promise<T>;
}


export async function getCharacters(): Promise<Character[]> {
  const response = await request<{ characters: Character[] }>("/api/characters");
  return response.characters;
}


export function getSession(): Promise<Session> {
  return request<Session>("/api/session");
}


export function startSession(characterId: string, voiceId: string): Promise<Session> {
  return request<Session>("/api/session", {
    method: "POST",
    body: JSON.stringify({ character_id: characterId, voice_id: voiceId }),
  });
}


export function endSession(sessionId: string): Promise<void> {
  return request<void>(`/api/session/${encodeURIComponent(sessionId)}`, { method: "DELETE" });
}

