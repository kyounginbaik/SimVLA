# Small examples

Start with the [CPU quickstart](../README.md#installation). To compose new task templates from the registered skills, see the [task authoring guide](../docs/task-authoring.md) and the seven validated templates in `src/simvla/templates/`. The goal files below are generated for particular kitchens; they are useful references, but they are not the editable task template format accepted by `simvla validate`.

- `goals/Isaac-Kitchen-v813-00.json`: bowl-to-drawer goals, including settings for the [one-demonstration command](../docs/simulation.md#4-simaction-generation).
- `goals/Isaac-Kitchen-v435-03.json`: goals for the recorded-action evaluation example.
- `*.reloadable.json`: authoring metadata associated with each task.
- `vqa/`: one captured frame from a successful GPU replay, with three RGB images, three segmentation images, and task metadata. This is a conversion example, not a training dataset.

From the checkout after installing the CPU package:

```bash
simvla vqa examples/vqa --output outputs/questions.jsonl
```

Expected result: 12 Q/A records. Paths in the generated records point to the bundled images on your machine. Keep those images available while using the output.

The goal files are **format examples and inputs for a separately provisioned simulator**. Their matching kitchen USDs and robot/object assets are not bundled, so the JSON files alone cannot replay or collect a demonstration.

To preflight a bundled generated goal on a CPU, run `simvla validate-goal examples/goals/Isaac-Kitchen-v813-00.json`. These bundled goals use the legacy format, so the command checks their structure and action channels; newly emitted version 2 goals also get skill/action checks.
