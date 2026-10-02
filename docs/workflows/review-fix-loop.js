const target = typeof workflowArgs.target === "string" ? workflowArgs.target : "the current change";
const maxRounds = 3;

function reportOf(run) {
  return run && run.result && run.result.report ? run.result.report : null;
}

const steps = {};
for (let round = 1; round <= maxRounds; round += 1) {
  const reviewStep = `review-${round}`;
  const review = await runs.run({
    step_key: reviewStep,
    agent: "reviewer",
    instruction: `Review ${target}. Return done when it is acceptable or blocked with concrete findings.`,
  });
  steps[reviewStep] = review.run_id;
  if (review.status !== "completed") {
    return {status: "failed", failed_step: reviewStep, steps};
  }

  const report = reportOf(review);
  if (!report) {
    return {status: "failed", failed_step: reviewStep, steps};
  }
  if (report.outcome === "blocked") {
    return {
      status: "completed",
      outcome: "blocked",
      reason: "review reported a blocked business condition",
      rounds: round,
      steps,
    };
  }
  if (report.outcome !== "done") {
    return {status: "failed", failed_step: reviewStep, rounds: round, steps};
  }
  const findings = Array.isArray(report.findings) ? report.findings : [];
  if (findings.length === 0) {
    return {
      status: "completed",
      outcome: "done",
      rounds: round,
      needs_fix: false,
      acceptance_guarantee: "No findings remain; this workflow does not guarantee acceptance.",
      steps,
    };
  }

  const fixStep = `fix-${round}`;
  const handoff = JSON.stringify(findings);
  if (handoff.length > 24000) throw Error("Findings exceed the instruction budget");
  const fix = await runs.run({
    step_key: fixStep,
    agent: "worker",
    instruction: `Fix these review findings for ${target}, then leave the workspace ready for another review:\n${handoff}`,
  });
  steps[fixStep] = fix.run_id;
  if (fix.status !== "completed") {
    return {status: "failed", failed_step: fixStep, rounds: round, steps};
  }
  const fixReport = reportOf(fix);
  if (!fixReport) {
    return {status: "failed", failed_step: fixStep, rounds: round, steps};
  }
  if (fixReport.outcome === "blocked") {
    return {
      status: "completed",
      outcome: "blocked",
      reason: "fix reported a blocked business condition",
      rounds: round,
      steps,
    };
  }
  if (fixReport.outcome !== "done") {
    return {status: "failed", failed_step: fixStep, rounds: round, steps};
  }
  if (round === maxRounds) {
    return {
      status: "completed",
      outcome: "blocked",
      reason: "review/fix limit reached; another review is required",
      rounds: round,
      steps,
    };
  }
}

return {
  status: "completed",
  outcome: "blocked",
  reason: "review/fix limit reached; another review is required",
  rounds: maxRounds,
  steps,
};
