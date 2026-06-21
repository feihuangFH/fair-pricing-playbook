# The Fair Pricing Playbook

[![License: CC BY 4.0](https://img.shields.io/badge/License-CC%20BY%204.0-lightgrey.svg)](https://creativecommons.org/licenses/by/4.0/)
[![DOI](https://img.shields.io/badge/DOI-10.2139%2Fssrn.6955039-blue)](https://doi.org/10.2139/ssrn.6955039)

A practical framework for Responsible AI in algorithmic pricing — four linked steps covering fairness definition, model design, welfare assessment, and post-deployment audit. Three reproducible case studies provide implementation evidence.

**Live site: [fair.feihuang.org](https://fair.feihuang.org)**

Developed by [Fei Huang](https://www.feihuang.org), UNSW Sydney.

## The four steps

| Step | Question | Key output |
|------|----------|------------|
| [1 · Define fairness](https://fair.feihuang.org/Playbook/Step-1-Define-Fairness.html) | What standard applies? | Documented criterion and scope |
| [2 · Design fair pricing](https://fair.feihuang.org/Playbook/Step-2-Design-Fair-Pricing.html) | How to build a model that meets it? | Fair cost model (MU / MDP / MCDP / MC) |
| [3 · Assess impact](https://fair.feihuang.org/Playbook/Step-3-Assess-Impact.html) | Who gains and loses once prices are set? | Welfare and profit analysis by group |
| [4 · Audit the system](https://fair.feihuang.org/Playbook/Step-4-Audit-the-System.html) | Does the deployed system pass? | Pass / fail / insufficient information |

## Case studies

| Case study | Topic | Tools |
|------------|-------|-------|
| [1 · Fair models](https://fair.feihuang.org/Case%20Study%201/case_study1.html) | Five anti-discrimination model designs on French motor data | R, GLM, XGBoost |
| [2 · Welfare implications](https://fair.feihuang.org/Case%20Study%202/case_study2.html) | Five pricing regulations evaluated for welfare and profit | Python, TensorFlow |
| [3 · Fairness testing](https://fair.feihuang.org/Case%20Study%203/case_study3.html) | CDP and proxy-discrimination audits with corrected inference on Illinois auto data | Python |

## Repository structure

```
Playbook/          Step pages (Quarto .qmd)
Case Study 1/      Fair cost-model design (R)
Case Study 2/      Welfare implications of pricing regulations (Python)
Case Study 3/      Fairness testing with corrected inference (Python)
index.qmd          Overview page
report.qmd         Standalone summary report
docs/              Rendered HTML (served by GitHub Pages)
```

## Reproduce locally

```bash
git clone https://github.com/feihuangFH/fair-pricing-playbook.git
cd fair-pricing-playbook
quarto render
quarto preview
```

Requires [Quarto](https://quarto.org/docs/get-started/), R (with `CASdatasets`, `xgboost`, `glmnet`), and Python (with `tensorflow`, `numpy`, `pandas`). Case Study 2 uses pre-computed optimisation outputs so Python re-execution is optional.

## How to cite

Huang, F. (2026). *The Fair Pricing Playbook: A practical framework for Responsible AI in algorithmic pricing*. SSRN. https://doi.org/10.2139/ssrn.6955039

A `CITATION.cff` file is included for one-click citation from the GitHub interface.

## License

Materials are licensed under [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/).
