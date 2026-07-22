import { expect, test } from "@playwright/test";


const characters = [
  {
    id: "gandalf",
    display_name: "Gandalf",
    summary: "A wise wandering wizard.",
    default_voice: "NATM1.pt",
    allowed_voices: ["NATM1.pt", "NATM2.pt"],
  },
  {
    id: "sauron",
    display_name: "Sauron",
    summary: "The calculating Dark Lord.",
    default_voice: "NATM3.pt",
    allowed_voices: ["NATM3.pt"],
  },
];


test("selects a character, waits without eviction, and ends the session", async ({ page }) => {
  let active = false;
  let deleted = false;
  await page.route("**/api/characters", (route) => route.fulfill({ json: { characters } }));
  await page.route("**/api/session", async (route) => {
    if (route.request().method() === "POST") {
      const body = route.request().postDataJSON();
      expect(body).toEqual({ character_id: "sauron", voice_id: "NATM3.pt" });
      active = true;
      await route.fulfill({
        status: 202,
        json: {
          state: "waiting_for_lease",
          session_id: "76697582-935e-4912-a24c-bdca71e00fae",
          character_id: "sauron",
          character_name: "Sauron",
          voice_id: "NATM3.pt",
          blocking_owner: { kind: "text", name: "coder" },
        },
      });
      return;
    }
    await route.fulfill({
      json: active
        ? {
            state: "waiting_for_lease",
            session_id: "76697582-935e-4912-a24c-bdca71e00fae",
            character_id: "sauron",
            character_name: "Sauron",
            voice_id: "NATM3.pt",
            blocking_owner: { kind: "text", name: "coder" },
          }
        : { state: "idle" },
    });
  });
  await page.route("**/api/session/76697582-935e-4912-a24c-bdca71e00fae", async (route) => {
    deleted = true;
    active = false;
    await route.fulfill({ status: 204, body: "" });
  });

  await page.goto("/");
  await page.getByRole("button", { name: "Select Sauron" }).click();
  await expect(page.getByLabel("Voice")).toHaveValue("NATM3.pt");
  if (process.env.VISUAL_QA === "1") {
    await page.waitForTimeout(750);
    await page.screenshot({ path: "/tmp/tolkien-voice-selector.png", fullPage: true });
  }
  await page.getByRole("button", { name: "Start conversation" }).click();
  await expect(page.getByRole("heading", { name: "Waiting for coder" })).toBeVisible();
  await expect(page.getByRole("button", { name: /evict|kill/i })).toHaveCount(0);
  await page.getByRole("button", { name: "Cancel" }).click();

  await expect(page.getByRole("heading", { name: "Whom would you speak with?" })).toBeVisible();
  expect(deleted).toBe(true);
});
