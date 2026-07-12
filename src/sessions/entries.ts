import { randomUUID } from "node:crypto";

import { z } from "zod";

import type { Message } from "../agent/index.js";

const toolCallSchema = z
  .object({ id: z.string(), name: z.string(), arguments: z.unknown() })
  .strict();
const toolErrorSchema = z.object({ code: z.string(), message: z.string() }).strict();
const messageSchema = z.discriminatedUnion("role", [
  z.object({ role: z.literal("user"), content: z.string() }).strict(),
  z
    .object({
      role: z.literal("assistant"),
      content: z.string(),
      toolCalls: z.array(toolCallSchema),
      providerMetadata: z.record(z.string(), z.unknown()).optional(),
    })
    .strict(),
  z
    .object({
      role: z.literal("tool"),
      toolCallId: z.string(),
      toolName: z.string(),
      ok: z.boolean(),
      content: z.string(),
      data: z.unknown().optional(),
      error: toolErrorSchema.optional(),
    })
    .strict(),
]);

const baseFields = {
  schemaVersion: z.literal(1),
  id: z.string().min(1),
  timestamp: z.string().datetime(),
};

export const sessionEntrySchema = z.discriminatedUnion("type", [
  z.object({ ...baseFields, type: z.literal("message"), message: messageSchema }).strict(),
  z.object({ ...baseFields, type: z.literal("model_change"), model: z.string().min(1) }).strict(),
  z
    .object({
      ...baseFields,
      type: z.literal("session_info"),
      sessionId: z.string().min(1),
      cwd: z.string().min(1),
      createdAt: z.string().datetime(),
    })
    .strict(),
]);

export type SessionEntry = z.infer<typeof sessionEntrySchema>;
export type MessageEntry = Extract<SessionEntry, { type: "message" }>;

function baseEntry() {
  return { schemaVersion: 1 as const, id: randomUUID(), timestamp: new Date().toISOString() };
}

export function createMessageEntry(message: Message): MessageEntry {
  return { ...baseEntry(), type: "message", message };
}

export function createModelChangeEntry(model: string): SessionEntry {
  return { ...baseEntry(), type: "model_change", model };
}

export function createSessionInfoEntry(sessionId: string, cwd: string): SessionEntry {
  const createdAt = new Date().toISOString();
  return { ...baseEntry(), type: "session_info", sessionId, cwd, createdAt };
}
