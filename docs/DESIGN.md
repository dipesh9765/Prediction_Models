# Design notes

The demo page uses the same tokens as the other web projects in this portfolio (`tokens.css`, copied
from `SIH-Big-Data-Searching/web/src`; the reasoning and the Hallmark rules it follows are written up in
that project's `web/DESIGN.md`).

- A tool, not a landing page: no hero, models are chosen from an underlined tab row, the answer is the
  one large figure, notes carry meaning in a left rule instead of a filled box.
- Prices and probabilities use a tabular monospace face so numbers line up; interval bounds are
  formatted the same way as the point estimate.
- The probability bar is a native `<meter>` with an accessible name, so it needs no inline style.
- `node scripts/check-design.mjs src/predictors/static` enforces the mechanical rules in CI.

Checked in a real browser against a stand-in API (tabs, a numeric and a probability answer, disabled
model, no console errors, no overflow at 1280px). Not checked here: the Python test suite, because this
machine's Application Control policy blocks scikit-learn's DLLs; CI runs it.
