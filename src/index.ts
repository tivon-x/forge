#!/usr/bin/env node

import { main } from "./cli/index.js";

const controller = new AbortController();
const cancel = () => controller.abort();

process.once("SIGINT", cancel);

try {
  process.exitCode = await main(process.argv, { signal: controller.signal });
} finally {
  process.removeListener("SIGINT", cancel);
}
