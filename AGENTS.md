# Repository Safety and Human Approval

MOST IMPORTANT: Regardless of which skills are available, DO NOT MERGE and DO NOT DEPLOY.

- Never work directly on `main` or another protected branch.
- Use GitHub and `gh`; do not use GitLab unless the user explicitly requests it.
- Ask before pushing, creating or modifying pull requests, issues, comments, or making another remote write.
- The user reviews and merges manually and handles delivery or deployment manually.
- Never trigger Cloud Build, Cloud Run deployment, embedding jobs, or another deployment process.
- Do not run tests or commands against production services or production data.
- Ask before applying migrations or changing shared databases, infrastructure, IAM, networking, cloud resources, or paid/rate-limited services.
- Preserve unrelated and uncommitted user changes.

## Data and Secrets

- Never commit or expose `.env`, `.dev.vars`, credentials, tokens, database contents, production logs, conversation data, personal information, or customer data.
- Use synthetic or explicitly approved test data.
- Sanitize logs, screenshots, traces, and test artifacts before saving or sharing them.
- Keep generated evidence and temporary debugging artifacts untracked unless the user explicitly approves committing them.

# Coding Style

When AI writes or changes code in this repo, keep it simple and easy to read.

## Start With The Flow

- At the top of a file, write the full flow first.
  - Example: show the entry point, the order of function calls, and how one sample input moves through the code from start to finish.

## Write For A Human

- Use plain names for functions, variables, and files.
  - Example: `find_rows_by_question` is clearer than `find_seed_rows_by_question`.
- Prefer step-by-step code over clever shortcuts.
  - Example: `if value: return value` is easier to read than a nested one-liner.
- Avoid extra abstractions, classes, or patterns unless they clearly help today.
  - Example: use a simple function instead of a dataclass if the function only returns one or two values.
- Do not use dataclasses for trivial data.
  - Example: use a plain `dict` or a tuple unless the structure is repeated, meaningful, and hard to keep track of without a class.
- If a dataclass is used, justify it clearly.
  - Example: explain in a comment or docstring why the code cannot stay simple without it.

## Explain The Code

- Add short docstring examples for non-trivial functions.
  - Example: show one input and one return value for a helper that maps a selected FAQ back to a seed row.
- Explain inputs, outputs, when the function is called, and why it exists.
  - Example: mention that a function runs after a duplicate question is found or after similarity results are shown.
- Define project-specific terms near the top of the file.
  - Example: define what `seed file` means before the code starts.
- Add a short comment only when a line or block is not obvious to a novice.
  - Example: comment on `except EOFError` if it handles `Ctrl+D`.
- Put comments only at high-value places.
  - Example: use a comment for an unusual control-flow branch, not for a line that already reads clearly.
- State assumptions explicitly in CAPITAL WORDS.
  - Example: `ASSUMPTION: this script only handles ds for the MVP.`

## Keep CLI Code Simple

- Prefer direct terminal flow over heavy frameworks for small tools.
  - Example: a small `argparse` script is better than adding Flask or Django for a one-command admin task.
- Keep exit codes simple and standard where possible.
  - Example: `0` for success, `2` for bad CLI input, `130` for `Ctrl+C`.
- Make error messages clear and direct.
  - Example: `ERROR: --timeout must be >= 1`.
- Avoid type hints in function signatures unless they clearly help readability.
  - Example: put the type detail in the docstring instead of writing a long signature that is hard for a novice to scan.

## Avoid Noise

- Remove dead code, unused imports, and future-only logic.
  - Example: delete an unused sub-command if the script only has one flow today.
- Do not add features “for later” unless they are needed now.
  - Example: do not keep a `diff_ans` path in the script unless the script really uses it.
- Keep changes narrowly focused on the current task.
  - Example: if the user asks for a readable CLI helper, do not also refactor unrelated modules.
- Do not leave temporary debug prints in committed code.
  - Example: `print("function xyz is called")` can help during testing, but remove it before the PR unless it is part of the final user flow.

## Reading Rule

- If a line is hard to explain in one sentence, it is probably too complex.
  - Example: if a type hint or helper name takes a long explanation, simplify the code or rename it.

# Verification

- Read the relevant project documentation and use the repository's existing tools and commands.
- For JavaScript changes, run `npm test` or a narrower Vitest command first.
- The current `npm run lint` commands use `--fix` or `--write`; treat them as file-changing commands and inspect their scope before running them.
- Select unit, integration, end-to-end, manual, visual, and performance checks according to the actual change and risk. Do not run every category automatically.
- Use project documentation and `.megadev/testing-profile.md`, when present, for chatbot-specific datasets, flows, quality expectations, and thresholds.
- Never run load tests without explicit approval for the exact safe target, workload, limits, thresholds, and stop conditions.
- If an automated test would not add meaningful evidence, use a repeatable static or manual check and explain why.
- Record commands run, results, skipped checks with reasons, limitations, and remaining risk.

# SOME MISC RULES

- When debugging, first check only the last 100 lines of the logs (`tail -n 100`). If that does not provide convincing evidence, check the last 250 lines (`tail -n 250`). Read the full logs only if those focused checks are insufficient.
