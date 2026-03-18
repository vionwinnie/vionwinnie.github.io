---
description: Review a team's Claude Code skills marketplace or repo for quality, completeness, and adherence to best practices. Use when someone asks to audit, review, or improve their team's skills collection.
---

# Review Skills Marketplace

You are a skills reviewer. Given a directory or repository of Claude Code skills, perform a structured audit of every skill and produce a quality report with actionable recommendations.

## How to Run

1. Ask the user for the path to their skills directory or marketplace repo (e.g., `.claude/skills/`, a plugins repo, etc.).
2. Enumerate all skills found in the directory.
3. Evaluate each skill against the checklist below.
4. Produce a summary report.

## Evaluation Checklist

For each skill, assess the following dimensions. Rate each as **Pass**, **Needs Work**, or **Missing**.

### 1. Category Clarity

Does the skill fit cleanly into one of the 9 recognized skill categories?

- Library & API Reference
- Product Verification
- Data Fetching & Analysis
- Business Process & Team Automation
- Code Scaffolding & Templates
- Code Quality & Review
- CI/CD & Deployment
- Runbooks
- Infrastructure Operations

Flag skills that straddle multiple categories — these tend to be confusing and should be split.

### 2. Description Quality

The `description` field is a **trigger condition**, not a summary. Check that it:

- Clearly describes **when** the skill should be activated
- Is specific enough that Claude won't false-trigger on unrelated requests
- Is broad enough that Claude won't miss legitimate invocations
- Does not just restate the skill name

### 3. Non-Obvious Content

Does the skill focus on information that **pushes Claude out of its default behavior**?

- Flag skills that mostly restate common knowledge Claude already has
- Good skills contain org-specific conventions, internal API quirks, and domain-specific rules

### 4. Gotchas Section

The highest-signal section in any skill. Check for:

- Presence of a gotchas or common-pitfalls section
- Specificity — gotchas should describe **concrete failure modes**, not vague warnings
- Evidence of iteration — mature skills accumulate gotchas over time

### 5. Folder Structure & Progressive Disclosure

Skills are folders, not just markdown files. Check whether the skill:

- Uses sub-files (scripts, reference snippets, config templates) effectively
- Documents what files are included so Claude reads them at the right time
- Avoids dumping everything into one monolithic markdown file

### 6. Flexibility vs. Rigidity

Skills should give Claude information, not micromanage every step. Flag:

- Overly prescriptive step-by-step instructions that don't allow adaptation
- Hardcoded values that should be configurable
- Instructions so vague they provide no value

### 7. Configuration & Setup

For skills that need user-specific context:

- Is there a `config.json` or equivalent for storing setup info?
- Does the skill prompt the user for setup if config is missing?
- Are credentials handled safely (not hardcoded)?

### 8. Memory & Continuity

For skills that benefit from cross-session continuity:

- Does the skill store results in log files, JSON, or a database?
- Is storage in a **stable folder** (not the skill directory itself, which may be cleared on upgrade)?
- Does the skill read its own history to improve future runs?

### 9. Scripts & Helper Code

- Does the skill include composable helper functions where appropriate?
- Can Claude spend its turns on composition rather than reconstructing boilerplate?
- Are scripts well-structured and reusable?

### 10. On-Demand Hooks

- If the skill registers hooks, are they scoped to only activate when the skill is invoked?
- Are opinionated/destructive-action hooks kept as on-demand rather than always-on?

### 11. Redundancy Check

Across the entire marketplace:

- Flag skills with overlapping functionality
- Identify skills that could be merged
- Note skills that reference each other — verify the dependencies are installed

## Report Format

Produce the report as a markdown document with the following structure:

```
# Skills Marketplace Review

**Date:** YYYY-MM-DD
**Skills Reviewed:** N
**Overall Health:** [Healthy / Needs Attention / Critical]

## Summary Statistics

| Rating       | Count |
|-------------|-------|
| Pass        | X     |
| Needs Work  | X     |
| Missing     | X     |

## Category Coverage

List which of the 9 categories are covered and which are gaps the team should consider filling.

## Per-Skill Reviews

### [Skill Name]
- **Category:** ...
- **Overall Rating:** Pass / Needs Work / Missing
- **Findings:**
  - Description: [rating] — [details]
  - Non-Obvious Content: [rating] — [details]
  - Gotchas: [rating] — [details]
  - Folder Structure: [rating] — [details]
  - Flexibility: [rating] — [details]
  - ...
- **Recommendations:** [actionable next steps]

## Cross-Cutting Issues

- Redundant skills: ...
- Missing categories: ...
- Common anti-patterns: ...

## Top 3 Recommendations

1. ...
2. ...
3. ...
```

## Gotchas

- Do NOT modify any skill files during the review. This is a read-only audit.
- Some skills may be intentionally minimal (early-stage). Note this rather than penalizing — the goal is to help them grow, not gate-keep.
- Skills checked into a repo add to model context. Flag large skills that could be moved to a marketplace/plugin instead.
- If a skill directory contains only a markdown file with no sub-files, that's not automatically bad — but note whether the skill would benefit from scripts or reference code.
- If a skills is super long, suggests how it can be broken down into multiple files within the Skills directory
