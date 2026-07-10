import { describe, expect, it } from "vitest";

import { VERSION } from "./version.js";

describe("VERSION", () => {
  it("matches the package version", () => {
    expect(VERSION).toBe("0.1.0");
  });
});
