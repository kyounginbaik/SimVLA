# Evaluation protocol

Use this protocol for comparable SimVLA policy results. The public repository currently provides
the evaluator and a recorded-action integration check, but no compatible policy checkpoint or
paper benchmark bundle.

## Result classes

Label every result as one of:

- **Paper result:** reported in the SimVLA paper and tied to its experiment configuration.
- **Public reproduction:** produced from published assets, checkpoint, command, and manifest.
- **Integration smoke:** checks loading, action application, rendering, and result writing. It is
  not policy evidence.

The results in [release validation](validation.md) are integration evidence unless explicitly
identified otherwise.

## Trial definition

A trial is one rollout identified by:

- task ID: `Isaac-Kitchen-v<kitchen_number>-<sub_number>`;
- embodiment and robot-model revision;
- checkpoint and normalization revision;
- kitchen, sub-variant, seed, and environment index;
- horizon, control rate, and action-chunk size.

Choose the task list, sub-variants, and seeds before running. A zero-shot result requires every
kitchen number in the evaluation set to be absent from policy training data. Keep all attempted
trials, including simulator, planning, timeout, and policy failures.

## Metrics

The primary metric is task success from the environment's physical success predicate:

```text
success rate = successful trials / attempted trials
```

Report the numerator and denominator beside every rate. Aggregate multiple tasks with the
unweighted mean of their task-level success rates, and also publish each task result. If task
progress is reported, define the ordered physical subgoals and scoring rule before evaluation;
the current public evaluator writes success rate but does not implement a canonical progress
metric.

## Command

Start with an unseen kitchen and fixed seeds:

```bash
KITCHEN_NUMBER=99001
SUB_NUMBER=00
simvla run evaluate --repo-root . -- \
  --task "Isaac-Kitchen-v${KITCHEN_NUMBER}-${SUB_NUMBER}" \
  --data <dataset-name> --task_language "<instruction>" \
  --model <policy-type> --host_ip <policy-server> \
  --num_envs 1 --horizon 400 --step_hz 20 --action_chunk 50 \
  --seeds 0 1 2 3 --OOD True --log_dir outputs/evaluation
```

Repeat the fixed seed list for each preselected sub-variant. The policy server, checkpoint,
preprocessing, and embodiment-matched normalization must remain unchanged across the comparison.

## Evidence to publish

For each result, retain:

| Field | Required evidence |
| --- | --- |
| Source | SimVLA commit and dirty-tree state |
| Runtime | OS, GPU, Python, Isaac Sim/Lab, Torch, cuRobo, and LeRobot versions |
| Inputs | Task IDs, asset hashes, checkpoint hash, normalization hash, and training-kitchen list |
| Protocol | Seeds, environment count, horizon, rate, action chunk, and success predicate |
| Outcomes | Per-trial status, successes/attempts, aggregate formula, logs, and videos |
| Failures | Timeout, simulator, planner, policy, and infrastructure failures without deletion |

Cluster job IDs are supplementary metadata; publish sanitized commands, logs, and hashes because
job IDs alone cannot be inspected outside the originating cluster.
