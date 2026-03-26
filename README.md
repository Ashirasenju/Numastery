# NumPy Mastery — 7-Day Project

A structured, hands-on curriculum to take you from NumPy basics to advanced mastery.
Each day has two files: **exercises** (with challenges to solve) and **tests** (pytest suite to validate your work).

## Structure

```
numpy-mastery/
├── day1_basics/          Array creation, dtypes, shapes
├── day2_indexing/        Slicing, fancy indexing, masking
├── day3_math/            Universal functions, broadcasting
├── day4_linear_algebra/  linalg, decompositions, solvers
├── day5_statistics/      Stats, distributions, random
├── day6_advanced/        Structured arrays, memory, C extensions
├── day7_capstone/        Real-world project tying it all together
└── requirements.txt
```

## Setup

```bash
pip install -r requirements.txt
```

## Daily Workflow

1. Open `dayN_*/exercises.py` — read the docstring, implement the function body
2. Run the test suite to check your work:
   ```bash
   pytest dayN_*/test_exercises.py -v
   ```
3. All tests green? Move to the next day.

## Difficulty Progression

| Day | Theme | Difficulty |
|-----|-------|-----------|
| 1 | Array Basics & dtypes | ⭐ |
| 2 | Indexing & Masking | ⭐⭐ |
| 3 | Math & Broadcasting | ⭐⭐⭐ |
| 4 | Linear Algebra | ⭐⭐⭐ |
| 5 | Statistics & Random | ⭐⭐⭐ |
| 6 | Advanced Internals | ⭐⭐⭐⭐ |
| 7 | Capstone Project | ⭐⭐⭐⭐⭐ |

## Tips
- Don't look at the tests before attempting the exercises — they reveal the expected output
- Use `np.info()` and `help()` liberally
- The official docs are your friend: https://numpy.org/doc/stable/
