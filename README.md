# NLPercep Analysis Pipeline

Analysis code for the paper *"Social Perception as Theory of Mind: Evidence from Annotator Disagreement in Sexism Detection"*.

This pipeline analyzes the [EXIST 2025](http://nlp.uned.es/exist2025/) dataset (tweets, memes, and TikTok videos) to study annotator disagreement through a Theory of Mind (ToM) lens. The core finding is a **detection-interpretation dissociation**: annotators converge on *whether* content is sexist but diverge on *what kind* of sexism they perceive and *what intent* they attribute to the author.

## Dataset Summary

|  | Tweets | Memes | TikToks |
| --- | --- | --- | --- |
| Total instances | 7,958 | 4,044 | 2,524 |
| English | 3,749 | 2,010 | 1,000 |
| Spanish | 4,209 | 2,034 | 1,524 |
| Unique annotators | 809 | 887 | 10 |
| Annotators per instance | 6 (3F + 3M) | 6 (3F + 3M) | 2-3 (variable gender) |

Each instance is annotated across three tasks (numbered by modality):

- **Task x.1** — Binary sexism detection (YES / NO)
- **Task x.2** — Source intention (DIRECT, REPORTED, JUDGEMENTAL, UNKNOWN)
- **Task x.3** — Sexism category (multi-label: IDEOLOGICAL-INEQUALITY, STEREOTYPING-DOMINANCE, OBJECTIFICATION, SEXUAL-VIOLENCE, MISOGYNY-NON-SEXUAL-VIOLENCE)

## Key Findings

### Detection-Interpretation Dissociation

| Metric | Tweets | Memes | TikToks |
| --- | --- | --- | --- |
| Detection entropy (Task x.1, mean) | 0.547 | 0.631 | 0.219 |
| Full agreement rate | 32.9% | 23.7% | 76.2% |
| Any disagreement | 67.1% | 76.3% | 23.8% |
| Intent entropy among YES (Task x.2) | 0.876 | 0.701 | 0.118 |
| Category Jaccard among YES (Task x.3) | 0.382 | 0.418 | 0.642 |

For tweets and memes, annotators who agree content is sexist still show near-maximum entropy on intent attribution and low category overlap. TikTok videos show much higher agreement overall, likely due to having only 2-3 annotators (from a pool of just 10) and the more explicit visual/audio modality.

### Gender Effects (with Effect Sizes and Holm-Bonferroni Correction)

#### Detection Level

| Metric | Tweets | Memes | TikToks |
| --- | --- | --- | --- |
| F YES rate | 0.450 | 0.586 | 0.472 |
| M YES rate | 0.461 | 0.528 | 0.451 |
| Difference (F-M) | -0.011 | +0.058 | +0.021 |
| Wilcoxon p | 0.142 (n.s.) | < 10^-24 *** | 0.104 (n.s.) |
| Rank-biserial r | -0.026 | **0.240** | 0.082 |

For memes, despite the extreme p-value, the effect size is **small** (r = 0.24). The 0.058 difference on a 0-1 scale is driven to significance by the large sample size (n = 4,044), not a large practical effect.

#### Categorization Level (Holm-corrected, Fisher's exact with Cohen's h)

| Category | Modality | F rate | M rate | Cohen's h | p (raw) | p (adjusted) |
| --- | --- | --- | --- | --- | --- | --- |
| MISOGYNY | Tweets | 0.235 | 0.216 | 0.045 | 0.001 | **0.005** |
| OBJECTIFICATION | Tweets | 0.285 | 0.273 | 0.026 | 0.053 | 0.159 (n.s.) |
| SEXUAL-VIOLENCE | Tweets | 0.184 | 0.172 | 0.033 | 0.016 | 0.064 (n.s.) |
| OBJECTIFICATION | Memes | 0.363 | 0.314 | **0.102** | 3.4 x 10^-9 | **1.7 x 10^-8** |
| OBJECTIFICATION | TikToks | 0.225 | 0.110 | **0.313** | 2.1 x 10^-15 | **1.0 x 10^-14** |
| IDEOLOGICAL-INEQ. | TikToks | 0.336 | 0.233 | **0.231** | 3.8 x 10^-9 | **1.5 x 10^-8** |

All tweet/meme category effects are small (|h| < 0.11). Video effects appear larger (h = 0.23-0.31) but are based on only 10 annotators, so individual preferences may dominate.

### Cross-Modal Annotator Pool

| Metric | Value |
| --- | --- |
| Tweets-Memes Jaccard similarity | 0.912 |
| All tweet annotators also in memes? | Yes (100%) |
| Memes-only annotators | 78 |
| Demographics match (gender, age, education) | Yes (all p > 0.4) |
| Demographics differ (ethnicity, country) | Small (V = 0.14, 0.25) |

Cross-modal differences are not confounded by annotator population differences.

## Project Structure

```
src/nlpercep/
    __main__.py              # Entry point (python -m nlpercep)
    run_analysis.py          # Main runner with CLI args
    data.py                  # Data loading and preprocessing
    correction.py            # Holm-Bonferroni correction, effect sizes (r, Cohen's h)
    section0_descriptive.py  # Descriptive statistics
    section1_disagreement.py # Agreement categories, entropy, Jaccard
    section2_intent_ambiguity.py  # Explicit vs implicit intent analysis
    section3_gender.py       # Gender moderation effects
    figures.py               # Publication-quality figures (PDF + PNG)
    qualitative.py           # Illustrative example selection
    extract_examples.py      # Targeted example extraction
    summary_table.py         # Paper summary table generation
    memes_analysis.py        # Parallel pipeline for memes dataset
    videos_analysis.py       # Parallel pipeline for TikTok videos (2-3 annotators)
    cross_modal_annotators.py  # Cross-modal annotator pool comparison
outputs/
    figures/                 # Generated figures (PDF + PNG, 300 DPI)
    tables/                  # Timestamped run directories with CSV/JSON results
    qualitative_examples.json  # Curated examples for the paper
    ANALYSIS_REPORT.md       # Comprehensive analysis report with all findings
```

## Setup

Requires Python >= 3.11 and [uv](https://docs.astral.sh/uv/).

```bash
uv sync
```

Place the EXIST 2025 dataset at `EXIST 2025 Dataset V0.1/` in the project root (not tracked by git).

## Usage

Run the full analysis pipeline:

```bash
uv run python -m nlpercep
```

Run specific sections only:

```bash
uv run python -m nlpercep --sections 0 1 2 3
uv run python -m nlpercep --sections 1 3 --no-memes --no-videos
```

Extract qualitative examples:

```bash
uv run python -m nlpercep.extract_examples
```

### CLI Options

| Flag | Description |
|---|---|
| `--sections 0 1 2 3` | Which analysis sections to run (default: all) |
| `--no-memes` | Skip memes dataset analysis |
| `--no-videos` | Skip TikTok videos dataset analysis |

## Outputs

Each run creates a timestamped directory under `outputs/tables/` containing:

- `full_log.txt` — Complete console output
- `s0_*.csv` — Descriptive statistics
- `s1_*.csv`, `s1_stats.json` — Agreement distributions and entropy
- `s2_*.json` — Intent ambiguity analysis
- `s3_*.json`, `s3_*.csv` — Gender effect analyses (with `p_adjusted` and `cohens_h` columns)
- `table2_summary.csv` — Paper summary table
- `cross_modal_annotators.json` — Annotator pool overlap and demographic comparison
- `memes/` — Parallel results for the memes dataset
- `videos/` — Parallel results for the TikTok videos dataset

Figures are saved to `outputs/figures/` as both PDF and PNG at 300 DPI, using the Wong (2011) colorblind-safe palette.

## Analysis Sections

- **Section 0**: Instance counts, demographics, label distributions
- **Section 1**: Agreement categories (unanimous YES through unanimous NO), Shannon entropy, conditional disagreement (intent entropy and Jaccard among majority-YES instances)
- **Section 2**: Explicit vs implicit intent — Mann-Whitney U test comparing detection entropy between groups (rank-biserial r for effect size)
- **Section 3**: Gender moderation — (a) overall F vs M detection rates (Wilcoxon signed-rank, rank-biserial r), (b) gender-aligned split analysis, (c) gender x ambiguity interaction (Mann-Whitney U, rank-biserial r), (d) per-category rates by gender (Fisher's exact, Holm-Bonferroni correction, Cohen's h)
- **Cross-Modal**: Annotator pool overlap (Jaccard), demographic comparison (chi-squared, Cramer's V), annotation volume per annotator

## Statistical Methods

| Method | Purpose | Effect Size |
| --- | --- | --- |
| Wilcoxon signed-rank | Paired F vs M detection rates | Rank-biserial r |
| Mann-Whitney U | Explicit vs implicit entropy; interaction test | Rank-biserial r |
| Fisher's exact test | Per-category gender rates (2x2 tables) | Cohen's h |
| Chi-squared | Demographic comparison across pools | Cramer's V |
| Holm-Bonferroni | Multiple comparison correction (5 tests per modality) | — |

Effect size benchmarks: rank-biserial r and Cohen's h use 0.1/0.3/0.5 (small/medium/large); Cramer's V uses 0.1/0.3/0.5.

## Dependencies

- pandas >= 2.1
- numpy >= 1.26
- scipy >= 1.11
- matplotlib >= 3.8
- seaborn >= 0.13

## Citation

If you use the EXIST 2025 dataset, please cite:

```bibtex
@inproceedings{10.1007/978-3-032-04354-2_16,
  author    = {Plaza, Laura and Carrillo-de-Albornoz, Jorge and Arcos, Iv\'{a}n and Rosso, Paolo and Spina, Damiano and Amig\'{o}, Enrique and Gonzalo, Julio and Morante, Roser},
  title     = {Overview of EXIST 2025: Learning with Disagreement for Sexism Identification and Characterization in Tweets, Memes, and TikTok Videos},
  year      = {2025},
  isbn      = {978-3-032-04353-5},
  publisher = {Springer-Verlag},
  address   = {Berlin, Heidelberg},
  url       = {https://doi.org/10.1007/978-3-032-04354-2_16},
  doi       = {10.1007/978-3-032-04354-2_16},
  booktitle = {Experimental IR Meets Multilinguality, Multimodality, and Interaction: 16th International Conference of the CLEF Association, CLEF 2025, Madrid, Spain, September 9--12, 2025, Proceedings},
  pages     = {266--289},
  numpages  = {24},
  keywords  = {sexism identification, sexism categorization, learning with disagreements, tweets, memes, TikTok videos, human-centric AI},
  location  = {Madrid, Spain}
}
```
