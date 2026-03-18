---
title: "Lessons from Building Claude Code: How Anthropic Uses Skills"
category: engineering
excerpt: A concise summary of Anthropic's internal lessons on building, distributing, and managing Claude Code Skills — from skill categories to best practices.
tags: claude-code ai-tools developer-productivity skills
comments: true
---

*This is a summary of [Thariq's thread on X](https://x.com/trq212/status/2033949937936085378) about how Anthropic uses Skills in Claude Code internally. All credit goes to the original author.*

---

## What Are Skills?

Skills are one of Claude Code's most popular extension points. A common misconception is that they're "just markdown files" — in reality, **skills are folders** that can include scripts, assets, data, and configuration options (including dynamic hooks). The best skills leverage this folder structure creatively.

## 9 Types of Skills

After cataloging hundreds of internal skills, Anthropic found they cluster into recurring categories:

1. **Library & API Reference** — How to correctly use a library, CLI, or SDK. Include code snippets and gotchas. *(e.g., internal billing library edge cases, design system guidance)*

2. **Product Verification** — Test and verify code using tools like Playwright or tmux. Worth investing serious engineering time here. Techniques include recording video of Claude's output and enforcing programmatic assertions at each step.

3. **Data Fetching & Analysis** — Connect to data/monitoring stacks with credentials, dashboard IDs, and common query workflows. *(e.g., funnel queries, Grafana lookups)*

4. **Business Process & Team Automation** — Automate repetitive workflows into single commands. Saving previous results in log files helps the model stay consistent across runs. *(e.g., standup posts, ticket creation, weekly recaps)*

5. **Code Scaffolding & Templates** — Generate framework boilerplate, especially useful when scaffolding has natural-language requirements beyond pure code.

6. **Code Quality & Review** — Enforce org code standards. Can include deterministic scripts and run automatically via hooks or GitHub Actions. *(e.g., adversarial review that spawns a subagent to critique code)*

7. **CI/CD & Deployment** — Fetch, push, deploy. *(e.g., babysit-pr: monitor PR → retry flaky CI → resolve conflicts → auto-merge)*

8. **Runbooks** — Take a symptom (alert, error signature), walk through investigation, produce a structured report.

9. **Infrastructure Operations** — Routine maintenance with guardrails for destructive actions. *(e.g., orphaned resource cleanup, dependency approval workflows)*

## Tips for Writing Good Skills

- **Don't state the obvious.** Claude already knows a lot about coding. Focus on information that pushes it *out* of its default thinking — your org's specific gotchas and conventions.

- **Build up Gotchas over time.** The highest-signal content in any skill is the gotchas section. Update it as Claude hits new edge cases.

- **Use the folder structure.** Think of the file system as progressive disclosure. Tell Claude what files are in your skill, and it will read them at appropriate times.

- **Be flexible, not rigid.** Since skills are reusable, avoid over-specifying. Give Claude the information it needs but let it adapt to the situation.

- **Store setup in config.** Skills that need user context (e.g., which Slack channel) should store setup info in a `config.json` that Claude can populate by asking the user.

- **Write good descriptions.** The description field is what Claude scans to decide if a skill matches a request — it's a trigger condition, not a summary.

- **Add memory via log files.** Skills can store data (text logs, JSON, even SQLite) to build continuity across runs. Store in a stable folder since skill directories may be cleared on upgrade.

- **Include scripts & helper code.** Giving Claude composable helper functions lets it spend its turns on *composition* rather than reconstructing boilerplate.

- **Use on-demand hooks.** Skills can register hooks that activate only when the skill is invoked. Great for opinionated behaviors you don't want running all the time. *(e.g., `/careful` blocks destructive commands, `/freeze` blocks edits outside a specific directory)*

## Distributing Skills

Two main approaches:

1. **Check into your repo** (under `.claude/skills/`) — works well for smaller teams
2. **Internal plugin marketplace** — scales better as skills multiply, since each checked-in skill adds to model context

For marketplaces: let skills emerge organically. Upload to a sandbox folder, share via Slack, and move to the marketplace once they gain traction. Curation before release is important — it's easy to create bad or redundant skills.

Skills can reference other skills by name for composition, though formal dependency management isn't built in yet.

## Measuring Skills

Use a `PreToolUse` hook to log skill usage across the org. This reveals which skills are popular and which are under-triggering relative to expectations.

## Key Takeaway

Skills are still early. Most of Anthropic's best skills started as a few lines and a single gotcha, then improved over time as Claude hit new edge cases. The best way to learn is to start building and iterating.

## Try It: A Skill That Reviews Your Skills

Want to put these principles into practice? I created a Claude Code skill that audits your team's skills marketplace against the best practices above — checking description quality, gotchas coverage, folder structure, redundancy, and more. It produces a structured report with actionable recommendations.

[View the review-skills-marketplace skill](/assets/skills/review-skills-marketplace.md)

---

*Original thread: [Thariq (@trq212) on X](https://x.com/trq212/status/2033949937936085378)*
