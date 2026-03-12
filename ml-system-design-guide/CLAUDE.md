# ML System Design Interview Guide

## Project Overview

Single-page HTML interview guide (`index.html`) with collapsible sections covering real-world ML systems. Dark-themed, responsive, uses Mermaid diagrams. Section source files live in `sections/` for reference; the canonical content is inlined in `index.html`.

## File Structure

```
index.html                          # Main guide (all content inlined)
sections/
  section-cursor.html               # Cursor code completion
  section-rag.html                  # Anthropic contextual RAG
  section-agents.html               # Anthropic multi-agent systems
  section-agent-design.html         # Agent design patterns (Lance Martin)
  section-moe.html                  # DeepMind Gemini MoE
  section-storybook.html            # Gemini Storybook multimodal
  section-nanobanana.html           # Nano Banana image generation
  section-waymo.html                # Waymo autonomous driving
assets/
  agent-design/                     # Blog images for agent design section
```

## HTML Structure & Conventions

### Adding a New Section

Every section follows this exact structure. Copy this template:

```html
<!-- In nav bar (inside <div class="nav-inner">) -->
<a href="#section-id">Section Name</a>

<!-- In systems container (inside <div id="systems-container">) -->
<div id="section-id">
<div class="system-section">
  <div class="section-header">
    <span class="company-badge badge-COMPANY">Company Name</span>
    <h2>System Title <span class="subtitle">— Short description</span></h2>
    <span class="toggle-icon">&#9660;</span>
  </div>
  <div class="section-content">

    <!-- 1. Architecture Overview (REQUIRED) -->
    <div class="subsection">
      <h3>Architecture Overview</h3>
      <p>Overview paragraph with key numbers and design philosophy.</p>
      <p style="margin-top: 0.5rem; font-size: 0.85rem; color: var(--text-secondary);">
        Based on <a href="URL" target="_blank" style="color: var(--accent-cyan);">Source Title</a>
      </p>
      <div class="diagram-container">
        <div class="mermaid">
graph TD
    <!-- Mermaid flowchart here -->
        </div>
      </div>
    </div>

    <!-- 2-N. Technical subsections (varies by topic) -->
    <div class="subsection">
      <h3>Subsection Title</h3>
      <p>Description paragraph.</p>
      <!-- Use <table class="comparison-table"> for structured comparisons -->
      <!-- Use <ul> with <li><strong>Term:</strong> explanation</li> for lists -->
    </div>

    <!-- Key Design Decisions (REQUIRED) -->
    <div class="subsection">
      <h3>Key Design Decisions</h3>
      <table class="comparison-table">
        <thead>
          <tr><th>Decision</th><th>Chosen Approach</th><th>Alternative</th><th>Rationale</th></tr>
        </thead>
        <tbody>
          <tr><td><strong>Decision name</strong></td><td>What was chosen</td><td>What else could work</td><td>Why this tradeoff</td></tr>
        </tbody>
      </table>
    </div>

    <!-- Interview Talking Points (REQUIRED) -->
    <div class="subsection">
      <div class="talking-points">
        <h4>Interview Talking Points</h4>
        <ul>
          <li><strong>Bold claim:</strong> Supporting explanation with specific numbers.</li>
        </ul>
      </div>
    </div>

    <!-- Sources (REQUIRED) -->
    <div class="sources">
      <strong style="color: var(--text-muted); font-size: 0.8rem;">Sources:</strong><br>
      <a href="URL" target="_blank">Source — Title (Date)</a>
    </div>

  </div>
</div>
</div>
```

### Checklist When Adding a Section

1. **Badge CSS** — Add `.badge-COMPANY` style in the `<style>` block (near line 177):
   ```css
   .badge-company { background: rgba(R, G, B, 0.15); color: var(--accent-COLOR); }
   ```
   Available accent colors: `--accent-blue`, `--accent-green`, `--accent-purple`, `--accent-orange`, `--accent-red`, `--accent-cyan`

2. **Nav link** — Add `<a href="#section-id">Name</a>` inside `<div class="nav-inner">`

3. **Hero count** — Update the system count in the hero `<p>` tag (e.g., "8 real-world ML systems")

4. **Section HTML** — Insert the section `<div id="section-id">...</div>` inside `<div id="systems-container">`, before the closing `</div>` and `<!-- Comparison Table -->` comment

5. **Comparison table** — Add a `<tr>` row to the `<section id="comparison">` table with: System, Company, Scale, Key Innovation, Latency Target, Core ML Task

6. **Section file** — Save a copy to `sections/section-NAME.html` (the section content only, without the outer `<div id="...">` wrapper)

### Mermaid Diagram Rules

- Use `graph TD` (top-down) for architecture diagrams
- Use `subgraph Name["Display Label"]` for grouping
- Avoid special characters in labels: no `→`, `°`, `<`, `>`, `&` — use plain text equivalents
- Apply dark theme styles to every subgraph:
  ```
  style SubgraphName fill:#1c2333,stroke:#58a6ff,stroke-width:2px,color:#e6edf3
  ```
- Diagrams render on page load via `renderAllDiagrams()` — sections are temporarily opened, diagrams rendered, then collapsed

### Content Style

- **Architecture Overview**: Lead with what the system does, key numbers, and core design philosophy
- **Tables**: Use `comparison-table` class. Column headers should be conceptual (Pattern / When to Use / Example), not generic (Column 1 / Column 2)
- **Lists**: `<strong>Term:</strong>` followed by explanation. Keep each bullet to 1-2 sentences
- **Talking Points**: 7-8 bullets. Each starts with a **bold claim** followed by the supporting argument with specific numbers. Write as if coaching someone for an interview — the bold text is what to say, the rest is why
- **Sources**: Link to original blog posts/papers with title and date
- **Key Design Decisions**: 5-6 rows minimum. Always frame as "Chosen vs. Alternative" with rationale explaining the tradeoff, not just why one is better
- **Quantitative**: Always include specific numbers (latency, accuracy, scale, cost reduction percentages)

## Workflow: Adding a New System

### Step 1: Research (use subagents in parallel)

Launch these agents simultaneously:

- **Web research agent**: Research the system's architecture, papers, blog posts, key metrics, and design decisions. Search for ML system design interview questions related to the topic.
- **Blog fetch agent** (if URL provided): Extract full content from the provided blog URL including all technical details, figures, and numbers.

### Step 2: Build the Section

Using research results:
1. Create `sections/section-NAME.html` with the full section content
2. Insert into `index.html` following the checklist above
3. Ensure Mermaid diagram uses plain text (no special chars)

### Step 3: Review (critic agent)

After adding the section, launch a **critic agent** to review. The critic should check:

**Structure compliance:**
- [ ] Has Architecture Overview with Mermaid diagram
- [ ] Has Key Design Decisions table (5+ rows, Chosen/Alternative/Rationale format)
- [ ] Has Interview Talking Points (7-8 bullets with bold claims + numbers)
- [ ] Has Sources section with linked references
- [ ] Badge CSS added, nav link added, hero count updated, comparison table row added

**Content quality:**
- [ ] Specific numbers cited (latency, accuracy, scale, cost)
- [ ] Design decisions frame genuine tradeoffs (not strawmen)
- [ ] Talking points are interview-ready (bold claim a candidate would say, rationale to back it up)
- [ ] No generic filler — every sentence adds information
- [ ] Source URLs are real and correctly attributed

**Technical accuracy:**
- [ ] Architecture diagram accurately represents the system
- [ ] Model names, benchmark numbers, and dates are correct
- [ ] Tradeoff rationales are technically sound
- [ ] No contradictions between subsections

**HTML/rendering:**
- [ ] Mermaid syntax has no special characters (no `→`, `°`, `<`, `>`, `&` in labels)
- [ ] All HTML entities properly escaped in table cells (`&lt;`, `&amp;`, `&deg;`)
- [ ] Section follows exact nesting: `div#id > div.system-section > div.section-header + div.section-content`
- [ ] `sections/section-NAME.html` file matches what's in `index.html`

### Agent Prompts

**Research agent prompt template:**
```
Research [SYSTEM/COMPANY]'s ML system design thoroughly for an interview guide. Find:
1. Architecture (components, models, pipeline)
2. Key technical papers and blog posts with specific results
3. Scale and infrastructure numbers
4. Design tradeoffs (with alternatives considered)
5. Common interview questions about this topic
6. Recent developments (2025-2026)
Return ALL findings with specific numbers and technical depth.
```

**Critic agent prompt template:**
```
Review the [SECTION NAME] section just added to /Users/manwaiwinnieyeung/Development/ml-system-design-guide/index.html.

Read the section in index.html and the corresponding sections/section-NAME.html file.
Check against every item in the CLAUDE.md review checklist under "Step 3: Review (critic agent)".

For each checklist item, report PASS or FAIL with specifics.
If any items FAIL, provide the exact fix needed (old text -> new text).
Do NOT make edits — only report findings.
```
