---
name: Analyst Precision
colors:
  surface: '#f7f9fb'
  surface-dim: '#d8dadc'
  surface-bright: '#f7f9fb'
  surface-container-lowest: '#ffffff'
  surface-container-low: '#f2f4f6'
  surface-container: '#eceef0'
  surface-container-high: '#e6e8ea'
  surface-container-highest: '#e0e3e5'
  on-surface: '#191c1e'
  on-surface-variant: '#424751'
  inverse-surface: '#2d3133'
  inverse-on-surface: '#eff1f3'
  outline: '#737783'
  outline-variant: '#c2c6d3'
  surface-tint: '#255dad'
  primary: '#00346f'
  on-primary: '#ffffff'
  primary-container: '#004a99'
  on-primary-container: '#9bbdff'
  inverse-primary: '#abc7ff'
  secondary: '#515f74'
  on-secondary: '#ffffff'
  secondary-container: '#d5e3fc'
  on-secondary-container: '#57657a'
  tertiary: '#5f2200'
  on-tertiary: '#ffffff'
  tertiary-container: '#833301'
  on-tertiary-container: '#ffa77e'
  error: '#ba1a1a'
  on-error: '#ffffff'
  error-container: '#ffdad6'
  on-error-container: '#93000a'
  primary-fixed: '#d7e2ff'
  primary-fixed-dim: '#abc7ff'
  on-primary-fixed: '#001b3f'
  on-primary-fixed-variant: '#00458f'
  secondary-fixed: '#d5e3fc'
  secondary-fixed-dim: '#b9c7df'
  on-secondary-fixed: '#0d1c2e'
  on-secondary-fixed-variant: '#3a485b'
  tertiary-fixed: '#ffdbcc'
  tertiary-fixed-dim: '#ffb694'
  on-tertiary-fixed: '#351000'
  on-tertiary-fixed-variant: '#7b2f00'
  background: '#f7f9fb'
  on-background: '#191c1e'
  surface-variant: '#e0e3e5'
  status-high-confidence: '#065F46'
  status-mid-confidence: '#92400E'
  status-low-confidence: '#991B1B'
  ui-border: '#CBD5E1'
  ui-text-primary: '#0F172A'
typography:
  display-lg:
    fontFamily: Hanken Grotesk
    fontSize: 30px
    fontWeight: '600'
    lineHeight: 38px
    letterSpacing: -0.02em
  headline-md:
    fontFamily: Hanken Grotesk
    fontSize: 20px
    fontWeight: '600'
    lineHeight: 28px
  headline-sm:
    fontFamily: Hanken Grotesk
    fontSize: 16px
    fontWeight: '600'
    lineHeight: 24px
  body-lg:
    fontFamily: Inter
    fontSize: 16px
    fontWeight: '400'
    lineHeight: 24px
  body-md:
    fontFamily: Inter
    fontSize: 14px
    fontWeight: '400'
    lineHeight: 20px
  label-md:
    fontFamily: Inter
    fontSize: 12px
    fontWeight: '600'
    lineHeight: 16px
    letterSpacing: 0.02em
  label-sm:
    fontFamily: Inter
    fontSize: 11px
    fontWeight: '500'
    lineHeight: 14px
  mono-data:
    fontFamily: Inter
    fontSize: 13px
    fontWeight: '400'
    lineHeight: 18px
rounded:
  sm: 0.125rem
  DEFAULT: 0.25rem
  md: 0.375rem
  lg: 0.5rem
  xl: 0.75rem
  full: 9999px
spacing:
  baseline: 4px
  gutter: 24px
  margin: 32px
  container-max-width: 1440px
---

## Brand & Style

This design system is engineered for high-stakes enterprise utility. It prioritizes cognitive clarity and speed of intent for internal analysts. The aesthetic is **Corporate / Modern** with a lean toward **Minimalism**, characterized by a rigorous grid, deliberate whitespace, and a complete absence of decorative flourishes.

The emotional response should be one of "controlled focus." By stripping away the conversational tropes of modern AI and the cluttered density of legacy enterprise software, the system creates a workspace that feels calm under time pressure. Every visual element serves a functional purpose: to assist the analyst in triaging data with speed and accuracy.

- **Focus:** Data-dense but readable.
- **Tone:** Professional, objective, and institutional.
- **Visual Strategy:** Structured layouts, distinct functional zones, and high-legibility typography.

## Colors

The palette is strictly professional and color-blind safe, relying on "Analytical Blue" and "Slate Grey" to create a neutral, non-distracting environment.

- **Primary Blue (#004A99):** Used for primary actions, active navigation states, and key interactive highlights.
- **Secondary Slate (#475569):** Used for supporting information, icons, and secondary actions.
- **Semantic Indicators:** Confidence levels are mapped to a high-contrast palette (Green, Amber, Red). To ensure accessibility, these colors must never be the sole indicator of meaning; they must always be accompanied by text labels or numerical values.
- **Surface Strategy:** A clean, near-white background reduces eye strain during long shifts, while subtle grey borders define functional zones without the need for heavy shadows.

## Typography

Typography is the primary vehicle for information hierarchy. **Hanken Grotesk** is used for structural headings to provide a modern, sharp look, while **Inter** is used for all functional and body text due to its exceptional legibility and systematic feel.

- **Hierarchy:** Large display sizes are avoided to maximize information density.
- **Consumer Narratives:** For the complaint text itself, use `body-lg` with a slightly wider line height to improve reading speed during triage.
- **Confidence Scores:** Use `label-md` in all-caps or bold weights to ensure these critical data points are easily scannable.
- **Contrast:** Ensure all text passes WCAG AA contrast ratios against their respective backgrounds.

## Layout & Spacing

The layout utilizes a **Fixed Grid** model on desktop to ensure a consistent scanning path for analysts. The screen is divided into clear functional regions: a persistent left-hand navigation/history panel and a main workspace split into a two-pane layout.

- **The Triage Workspace:** The left pane contains the input (complaint narrative); the right pane contains the model output (predictions and confidence).
- **Responsive Behavior:** On smaller screens (tablets), the panes stack vertically to maintain the 16px font size legibility.
- **Spacing Rhythm:** A 4px baseline grid ensures consistent vertical rhythm. Components are separated by 24px (gutter) to prevent visual crowding.
- **Margins:** Generous 32px outer margins ensure the content feels contained and professional.

## Elevation & Depth

This design system avoids heavy shadows and physical metaphors to maintain a "flat," digital-first aesthetic. Depth is communicated through **Tonal Layers** and **Low-Contrast Outlines**.

- **Surface Levels:**
  - Level 0 (Background): #F8FAFC.
  - Level 1 (Cards/Panels): #FFFFFF with a 1px border (#CBD5E1).
  - Level 2 (Active/Hover): A very soft, 2px blur shadow only to indicate interactivity.
- **Focus States:** High-visibility 2px solid blue outlines for keyboard navigation, ensuring analysts can operate the tool without a mouse.
- **Separation:** Vertical and horizontal dividers are used to separate the input area from the prediction area, reinforcing the "decision support" nature of the tool.

## Shapes

The shape language is **Soft (0.25rem)**, providing just enough approachability to feel modern without losing the "institutional" precision of sharp-edged enterprise software.

- **Primary Elements:** Buttons, input fields, and tags use the base 4px (0.25rem) radius.
- **Large Containers:** Main cards and the "Triage Panel" use 8px (0.5rem) to subtly distinguish them from the atomic elements within them.
- **Data Tags:** Confidence indicators and category labels use the standard 4px radius; avoid pill-shapes to maintain a professional, non-consumer look.

## Components

### Buttons
- **Primary:** Solid Blue (#004A99), white text. Used for "Confirm Classification" or "Submit."
- **Secondary:** White background, Slate border (#CBD5E1), Slate text. Used for "Reset" or "Save Draft."
- **Ghost:** No border, blue text. Used for minor actions like "Copy Narrative."

### Prediction Cards
The core of the UI. Must display:
- **Category Name:** Bold `headline-sm`.
- **Confidence Score:** Percentage text paired with a semantic color bar (not just a bar).
- **Alternative Routes:** A list of the next 2-3 likely categories with smaller text.

### Input Fields
- **Narrative Box:** Large text area with a subtle background tint (#F1F5F9) to indicate it is the source data.
- **Category Picker:** Searchable dropdown with "Model Suggested" categories pinned to the top.

### Chips & Tags
- Used for confidence levels (High, Med, Low).
- Rectangular with slight rounding; no icons. Text-based labels are mandatory.

### Lists
- History of previous triages should be a clean, high-density list with clear timestamps and the assigned category.
- Use zebra-striping (alternating #F8FAFC and #FFFFFF) for long lists to assist horizontal tracking.
