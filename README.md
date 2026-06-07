The Fair Pricing Playbook
===========

A four-step pathway for fair algorithmic pricing: define fairness → design fair pricing → assess welfare impact → audit the system. Three reproducible insurance case studies supplement the pathway for technical readers.

Developed by [Fei Huang](https://www.feihuang.org) at UNSW Sydney.

Please open the site at **[fair.feihuang.org](https://fair.feihuang.org)** (source: [github.com/feihuangFH/fair-pricing-playbook](https://github.com/feihuangFH/fair-pricing-playbook)).

## Pathway

1. [Define fairness](https://fair.feihuang.org/Playbook/Step-1-Define-Fairness.html)
2. [Design fair pricing](https://fair.feihuang.org/Playbook/Step-2-Design-Fair-Pricing.html)
3. [Assess impact](https://fair.feihuang.org/Playbook/Step-3-Assess-Impact.html)
4. [Audit the system](https://fair.feihuang.org/Playbook/Step-4-Audit-the-System.html)

## Case Studies (technical)

- [Overview on home page](https://fair.feihuang.org/index.html#case-studies)
- [1: Fair cost models](https://fair.feihuang.org/Case%20Study%201/case_study1.html)
- [2: Welfare implications](https://fair.feihuang.org/Case%20Study%202/case_study2.html)
- [3: Fairness testing](https://fair.feihuang.org/Case%20Study%203/case_study3.html)

## Local development

```bash
cd "/Users/feihuang/Dropbox/Fei & Teaching/The Fair Pricing Playbook"
quarto render
quarto preview
```

## Publish to GitHub Pages

```bash
quarto render
git add .
git commit -m "Update site"
git push
```

GitHub Pages should serve the `docs/` folder on the `main` branch. Set the custom domain to `fair.feihuang.org` in the repository Pages settings.

## How to cite

Huang, F. (2026). *The Fair Pricing Playbook: A practical framework for fair algorithmic pricing*. [fair.feihuang.org](https://fair.feihuang.org) (source: [github.com/feihuangFH/fair-pricing-playbook](https://github.com/feihuangFH/fair-pricing-playbook)).

## License

Materials are licensed under [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/). See LICENSE.
