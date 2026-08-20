# Colorblind Accessibility — CV Research Project

Simulates color vision deficiency (CVD), quantifies image/UI accessibility
for CVD users, and applies conditional correction when accessibility is
below a threshold.

## Setup

```bash
python -m venv venv
source venv/bin/activate   # or venv\Scripts\activate on Windows
pip install -r requirements.txt
```

## Structure

- `src/color_spaces.py` — RGB <-> LMS conversion
- `src/simulate.py` — Brettel/Machado CVD simulation
- `src/metrics.py` — accessibility scoring
- `src/correct.py` — daltonization / PyTorch-based correction
- `src/utils.py` — image I/O helpers
- `notebooks/` — experiments and visual comparisons
- `research_log.md` — running research notes and decisions
