const target = typeof workflowArgs.target === "string" ? workflowArgs.target : "the current change";

function reportOf(run) {
  return run && run.result && run.result.report ? run.result.report : null;
}

const settled = await Promise.allSettled([
  runs.run({
    step_key: "review-security",
    agent: "reviewer",
    instruction: `Review ${target} for security and boundary regressions.`,
  }),
  runs.run({
    step_key: "review-tests",
    agent: "reviewer",
    instruction: `Review ${target} for missing failure and cancellation coverage.`,
  }),
]);

const findings = [];
const steps = {};
let failed = false;
let blocked = false;
const stepKeys = ["review-security", "review-tests"];
for (let index = 0; index < settled.length; index += 1) {
  const item = settled[index];
  if (item.status === "rejected") {
    failed = true;
    continue;
  }
  const run = item.value;
  steps[stepKeys[index]] = run.run_id;
  if (run.status !== "completed") {
    failed = true;
    continue;
  }
  const report = reportOf(run);
  if (!report) {
    failed = true;
    continue;
  }
  if (report.outcome === "blocked") {
    blocked = true;
  } else if (report.outcome !== "done") {
    failed = true;
    continue;
  }
  if (Array.isArray(report.findings)) {
    for (const finding of report.findings) {
      if (typeof finding === "string" && findings.length < 32) {
        findings.push(finding);
      }
    }
  }
}

return {
  status: failed ? "failed" : "completed",
  outcome: failed ? null : blocked ? "blocked" : "done",
  needs_fix: !failed && findings.length > 0,
  acceptance_guarantee: "Review findings are advisory; done is not an acceptance guarantee.",
  findings,
  steps,
};
