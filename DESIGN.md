---
name: Skynet_DS Review Console
description: A dark-first observability console for ML pull-request review and drift triage, with a per-detector drift matrix as its signature.
colors:
  bg: "oklch(0.17 0.02 252)"
  rail: "oklch(0.145 0.022 252)"
  surface: "oklch(0.205 0.024 252)"
  surface-2: "oklch(0.245 0.027 252)"
  line: "oklch(0.32 0.03 252)"
  line-soft: "oklch(0.265 0.027 252)"
  text: "oklch(0.95 0.008 252)"
  text-2: "oklch(0.77 0.02 252)"
  text-3: "oklch(0.66 0.025 252)"
  accent: "oklch(0.8 0.12 190)"
  accent-ink: "oklch(0.2 0.04 200)"
  crit: "oklch(0.7 0.21 25)"
  high: "oklch(0.75 0.17 55)"
  med: "oklch(0.84 0.14 92)"
  low: "oklch(0.77 0.1 235)"
  info: "oklch(0.7 0.03 252)"
  pass: "oklch(0.8 0.16 155)"
  warn: "oklch(0.84 0.14 92)"
  light-bg: "oklch(0.973 0.006 252)"
  light-surface: "oklch(0.995 0.003 252)"
  light-text: "oklch(0.22 0.03 252)"
  light-accent: "oklch(0.5 0.1 195)"
  light-crit: "oklch(0.5 0.22 25)"
typography:
  title:
    fontFamily: "system-ui, -apple-system, 'Segoe UI', Roboto, 'Helvetica Neue', sans-serif"
    fontSize: "20px"
    fontWeight: 600
    lineHeight: 1.3
    letterSpacing: "-0.015em"
  kpi:
    fontFamily: "system-ui, -apple-system, 'Segoe UI', Roboto, 'Helvetica Neue', sans-serif"
    fontSize: "22px"
    fontWeight: 600
    lineHeight: 1.15
    letterSpacing: "-0.02em"
  headline:
    fontFamily: "system-ui, -apple-system, 'Segoe UI', Roboto, 'Helvetica Neue', sans-serif"
    fontSize: "14px"
    fontWeight: 600
    lineHeight: 1.5
  body:
    fontFamily: "system-ui, -apple-system, 'Segoe UI', Roboto, 'Helvetica Neue', sans-serif"
    fontSize: "14px"
    fontWeight: 400
    lineHeight: 1.5
    fontFeature: "tabular-nums"
  label:
    fontFamily: "system-ui, -apple-system, 'Segoe UI', Roboto, 'Helvetica Neue', sans-serif"
    fontSize: "12px"
    fontWeight: 500
    lineHeight: 1.5
  mono:
    fontFamily: "ui-monospace, 'Cascadia Mono', 'SF Mono', Menlo, Consolas, monospace"
    fontSize: "12.5px"
    fontWeight: 400
    lineHeight: 1.5
rounded:
  sm: "4px"
  md: "6px"
  pill: "11px"
spacing:
  xs: "6px"
  sm: "8px"
  md: "12px"
  lg: "16px"
  xl: "24px"
components:
  panel:
    backgroundColor: "{colors.surface}"
    rounded: "{rounded.md}"
    padding: "12px 16px"
  kpi-tile:
    backgroundColor: "{colors.surface}"
    rounded: "{rounded.md}"
    padding: "14px 16px"
  chip:
    textColor: "{colors.info}"
    rounded: "{rounded.pill}"
    height: "22px"
    padding: "0 8px"
  filter:
    textColor: "{colors.text-2}"
    rounded: "13px"
    height: "26px"
    padding: "0 10px"
  rail-link:
    textColor: "{colors.text-2}"
    rounded: "{rounded.md}"
    padding: "7px 8px"
  rail-link-active:
    backgroundColor: "{colors.surface-2}"
    textColor: "{colors.text}"
  matrix-cell-hit:
    textColor: "{colors.crit}"
    rounded: "{rounded.sm}"
    height: "24px"
    typography: "{typography.mono}"
  matrix-cell-ok:
    backgroundColor: "{colors.surface-2}"
    textColor: "{colors.text-2}"
    rounded: "{rounded.sm}"
    height: "24px"
---

# Design System: Skynet_DS Review Console

## Overview

**Creative North Star: "The Instrument Panel"**

A working observability console in the lineage of Langfuse, LangSmith, Helicone and Phoenix, not a report page. A fixed left rail, a thin top bar, dense tables and state chips on a steel-navy ground. Every neutral is tinted toward hue 252 so nothing is pure grey; one teal accent marks selection and the brand; colour otherwise appears only as state (severity, pass, fail, live, sample).

The signature is the drift detector matrix: columns as rows, detection methods as columns, each cell the method's own statistic, filled red only where that detector fired. Everything else is quiet so the matrix and the severity colours carry the signal. Dark is the default; a cool-white light theme reuses the same token names and hues.

**Key Characteristics:**
- Dark-first, two themes, one set of token names (light overrides under the root `data-theme="light"` attribute).
- Hairline borders and tonal layering; no shadows, no glows.
- System sans for prose and UI, monospace for paths, feature names, statistics; tabular figures everywhere.
- Dense: 40px table rows, 13px table text, 12-12.5px meta.
- Bars show real proportions only.

## Colors

Cool steel-navy neutrals, one teal accent, and a separate state palette that never doubles as decoration.

### Primary
- **Signal Teal** (`accent`; dark oklch(0.8 0.12 190), light oklch(0.5 0.1 195)): current nav item icon, brand mark, focus ring, default bar fill, text selection tint. Nothing else.
- **Teal Ink** (`accent-ink`): text on solid accent fills.

### Neutral
- **Deep Steel** (`bg`), **Rail Night** (`rail`, sidebar, one step darker), **Panel Slate** (`surface`) and **Raised Slate** (`surface-2`, hover, table header, expanded rows, bar tracks): the four-step tonal ladder.
- **Hairline** (`line`) for input-like borders and the fix box; **Soft Hairline** (`line-soft`) for every panel, table and rail divider.
- **Text / Text 2 / Text 3** (`text`, `text-2`, `text-3`): primary, secondary, tertiary and meta copy. Light theme text-3 is oklch(0.5 0.03 252) to keep contrast on white.

### State roles
Severity: `crit` red, `high` orange, `med` yellow, `low` blue, `info` desaturated grey-blue. Outcome: `pass` green, `crit` doubles as fail. Provenance: `pass` for LIVE, `warn` (same yellow as `med`) for SAMPLE. Direction of change in attribution: increased uses `high`, decreased uses `low`, reshaped is neutral. Light theme darkens every state to roughly L 0.46-0.52 so text on tinted chip backgrounds stays legible.

### Named Rules
**The State-Only Colour Rule.** Hue outside the 252 shell family means a state. If a coloured element does not encode severity, outcome, provenance or direction, it is teal or neutral.
**The Tint-Not-Fill Rule.** State appears as coloured text on a 14-15% tint of itself with a 28-32% border, not as a solid block. Solid fills are reserved for bar segments.

## Typography

**Display/Body Font:** the platform system sans (system-ui, with Segoe UI, Roboto, Helvetica Neue fallbacks)
**Mono Font:** ui-monospace (Cascadia Mono, SF Mono, Menlo, Consolas fallback)

**Character:** Operating-system native and unbranded so the data is the voice; mono marks anything a human might copy (paths, feature names, statistics).

### Hierarchy
- **Title** (600, 20px, 1.3, -0.015em): the PR title, one per page.
- **KPI value** (600, 22px, 1.15, -0.02em; 18px under 560px): the four headline figures.
- **Headline** (600, 14px): panel headings.
- **Body** (400, 14px, 1.5): base size; table text 13px; detail copy capped at 78ch.
- **Label** (500, 12-12.5px): table headers, chips, filters, KPI labels, legends; sentence case, never uppercase, no tracking.
- **Mono** (12-13px): locations, subjects, feature names, matrix cells.

### Named Rules
**The Mono-For-Copyables Rule.** Paths, column names and statistics are monospace; prose never is.

## Layout

A two-column shell: 220px sticky rail plus a fluid main column, with a 48px sticky blurred top bar and content padded 24px, max 1480px. Overview opens with four equal KPI tiles (12px gap). Below, a 1.65fr / 1fr split puts Findings beside Root cause; the full-width Data drift panel follows. Spacing rhythm is 6/8/12/16/24px.

Responsive: at 1180px KPIs go 2-up and the split stacks. At 860px the shell becomes one column; the rail turns into a static top block with the brand and theme toggle on one row (toggle label hidden, icon only) and the four nav items on a second row, all reachable without scrolling; the top bar goes static and the root breadcrumb is hidden. At 560px the Location column is dropped from the findings table and its path moves into the expanded detail row; nav icons are hidden; KPI text shrinks. At 359px KPIs stack to one column. Tables scroll horizontally inside their panel rather than the page.

## Elevation & Depth

Flat. Depth is tonal (bg, surface, surface-2) plus hairline borders. There are no box-shadows. The only blur is the top bar's backdrop blur over a 92% bg tint. Hover and expansion change background to surface-2.

### Named Rules
**The Hairline-Not-Shadow Rule.** Separation comes from a 1px line-soft border or a tonal step, never from a shadow or glow.

## Shapes

Quiet geometry: 6px radius on panels, tiles, nav links, the fix box; 4px on matrix cells, code and focus outlines; fully rounded pills on chips (22px tall) and filters (26px tall); 6px bars with 3px ends. Chips carry a 6px state dot; filters carry a 7px rounded-square swatch. Icons are 16px, 1.5px stroke, inline SVG with round caps. No raster assets ship.

## Components

### Chips (state)
Pill, tinted text on 14% self-tint with 28% border and leading dot. Variants set by one `--c` variable: severity (CRITICAL to INFO), pass/fail, live/sample. `plain` drops the dot for neutral labels.

### Filters
Severity toggle pills with a square swatch and count; pressed state is a 16% tint, 45% border, full-contrast text.

### KPI tiles
Surface tile, 12.5px label with optional chip, 22px value, 12.5px foot. Bad verdict values turn crit, good turn pass. Tiles hold bars: severity mix, drift share and AUC against threshold.

### Bars
**Real proportions only.** A bar's width is a measured share or value. Segmented bars (severity mix, attribution share) are flex segments sized by count or share with a 3px minimum so non-zero values stay visible; threshold bars are a single fill with a 2px tick marking the threshold. They grow in once (700ms) on load. No sparklines, no trend lines, no decorative bars; the data has no time series.

### Data tables
13px, 40px rows, surface-2 header in 12px text-3, soft-hairline row rules, surface-2 row hover. Finding rows expand via a chevron toggle (rotates 90deg) into a surface-2 detail row with prose and a bordered Suggested fix box.

### Detector matrix (signature)
Rows are dataset columns (mono name), columns are detection methods plus a sample-count column. Each cell is a 56px-min, 24px mono pill showing the method's statistic: **hit** (crit text, 15% crit tint, 32% border, weight 600), **ok** (text-2 on surface-2), **na** (an em dash for not run or skipped, with the reason in the tooltip and aria-label). A drifted row's name turns crit. A legend under the table repeats the three cell states and explains that values are each method's statistic.

### Navigation
Rail links are 7px 8px, text-2 with a 16px icon; hover and current (`aria-current`) use surface-2 and full text, the current icon turns teal. Theme toggle pins to the rail bottom (a bordered button; sun icon in dark, moon in light) and the choice persists to localStorage, applied before first paint.

### Provenance chips
Every data panel header and the top bar carry a LIVE or SAMPLE chip; sample data is never unlabelled.

### Motion
Ease `cubic-bezier(0.16, 1, 0.3, 1)`; 120ms colour transitions, 160ms chevron rotation, 700ms bar grow. All disabled under `prefers-reduced-motion`.

## Do's and Don'ts

### Do:
- **Do** keep neutrals tinted to hue 252 and change themes by overriding the same token names.
- **Do** encode every colour as a state, and tint rather than fill.
- **Do** size every bar from real data; mark thresholds with the 2px tick.
- **Do** put paths, feature names and statistics in mono with tabular figures.
- **Do** keep a visible focus ring (2px accent, 2px offset) on every control.
- **Do** label every panel LIVE or SAMPLE.

### Don't:
- **Don't** add shadows, glows, or gradient fills; use hairlines and tonal steps.
- **Don't** draw trend lines, sparklines, or any bar that does not map to a measured value.
- **Don't** use teal for state, or a state colour for decoration.
- **Don't** use raster images, emoji, or glyph characters as icons; icons are inline SVG.
- **Don't** drop to a centred card-stack or hero layout; the console shell is the form.
