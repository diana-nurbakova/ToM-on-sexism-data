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

### Gender Effects

| Effect | Tweets | Memes | TikToks |
| --- | --- | --- | --- |
| F vs M detection (Wilcoxon p) | 0.142 (n.s.) | < 0.001 *** | 0.104 (n.s.) |
| F YES rate | 0.450 | 0.586 | 0.472 |
| M YES rate | 0.461 | 0.528 | 0.451 |
| OBJECTIFICATION gender gap (Fisher p) | 0.053 | < 0.001 *** | < 0.001 *** |
| IDEOLOGICAL-INEQUALITY gender gap (Fisher p) | 0.422 | 0.602 | < 0.001 *** |

For tweets, gender does not predict detection but shapes categorization (F annotators assign harm categories more often). For memes, gender predicts both detection and categorization, with a strong objectification gap. For TikToks, gender does not predict detection but F annotators assign OBJECTIFICATION (OR=2.36) and IDEOLOGICAL-INEQUALITY (OR=1.67) at significantly higher rates.

## Project Structure

```
src/nlpercep/
    __main__.py              # Entry point (python -m nlpercep)
    run_analysis.py          # Main runner with CLI args
    data.py                  # Data loading and preprocessing
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
outputs/
    figures/                 # Generated figures (PDF + PNG, 300 DPI)
    tables/                  # Timestamped run directories with CSV/JSON results
    qualitative_examples.json  # Curated examples for the paper
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
- `s3_*.json`, `s3_*.csv` — Gender effect analyses
- `table2_summary.csv` — Paper summary table
- `memes/` — Parallel results for the memes dataset
- `videos/` — Parallel results for the TikTok videos dataset

Figures are saved to `outputs/figures/` as both PDF and PNG at 300 DPI, using the Wong (2011) colorblind-safe palette.

## Analysis Sections

- **Section 0**: Instance counts, demographics, label distributions
- **Section 1**: Agreement categories (unanimous YES through unanimous NO), Shannon entropy, conditional disagreement (intent entropy and Jaccard among majority-YES instances)
- **Section 2**: Explicit vs implicit intent — Mann-Whitney U test comparing detection entropy between groups
- **Section 3**: Gender moderation — (a) overall F vs M detection rates (Wilcoxon), (b) gender-aligned split analysis, (c) gender x ambiguity interaction, (d) per-category rates by gender (Fisher's exact test)

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
