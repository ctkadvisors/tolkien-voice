import { WSMessage } from "./types";

export const shouldFlushPlayback = (message: WSMessage): boolean =>
  message.type === "control" && message.action === "bargeIn";
