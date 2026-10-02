const target = typeof workflowArgs.target === "string" ? workflowArgs.target : "the current change";

function reportOf(run) {
  return run && run.result && run.result.report ? run.result.report : null;
}

function resultForInstruction(run) {
  const report = reportOf(run);
  const result = report || (run && run.result && run.result.final_output);
  const text = typeof result === "string" ? result : JSON.stringify(result || {});
  if (text.length > 24000) throw Error("Handoff exceeds the instruction budget");
  return text;
}

const scout = await runs.run({
  step_key: "scout",
  agent: "scout",
  instruction: `Inspect ${target} and identify the files and risks a worker should handle.`,
});
if (scout.status !== "completed") {
  return {status: "failed", failed_step: "scout", run_id: scout.run_id};
}
if (reportOf(scout)?.outcome === "blocked") {
  return {status: "completed", outcome: "blocked", report: reportOf(scout), steps: {scout: scout.run_id}};
}

const worker = await runs.run({
  step_key: "worker",
  agent: "worker",
  instruction: `Implement the bounded change for ${target}. Scout result:\n${resultForInstruction(scout)}`,
});
if (worker.status !== "completed") {
  return {status: "failed", failed_step: "worker", run_id: worker.run_id};
}
if (reportOf(worker)?.outcome === "blocked") {
  return {
    status: "completed", outcome: "blocked", report: reportOf(worker),
    steps: {scout: scout.run_id, worker: worker.run_id},
  };
}

const reviewer = await runs.run({
  step_key: "reviewer",
  agent: "reviewer",
  instruction: `Review the implementation for ${target}; return a task-report with actionable findings.\nWorker result:\n${resultForInstruction(worker)}`,
});
if (reviewer.status !== "completed") {
  return {status: "failed", failed_step: "reviewer", run_id: reviewer.run_id};
}

const report = reportOf(reviewer);
if (!report) {
  return {status: "failed", failed_step: "reviewer", run_id: reviewer.run_id};
}
return {
  status: "completed",
  outcome: report.outcome,
  needs_fix: Array.isArray(report.findings) && report.findings.length > 0,
  acceptance_guarantee: "Review findings are advisory; done is not an acceptance guarantee.",
  steps: {
    scout: scout.run_id,
    worker: worker.run_id,
    reviewer: reviewer.run_id,
  },
};
