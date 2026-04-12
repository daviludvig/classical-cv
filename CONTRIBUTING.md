# Contributing

## Branching rules

Direct pushes to `main` are blocked. Every change must go through a Pull Request.

### Workflow

```bash
# 1. Make sure main is up to date
git checkout main
git pull origin main

# 2. Create a branch
git checkout -b <type>/<short-description>

# 3. Work, commit
git add <files>
git commit -m "type: description"

# 4. Push and open a PR to main
git push origin <type>/<short-description>
```

Then open a Pull Request on GitHub targeting `main`.

### Branch naming

| Prefix | Use for |
| ------ | ------- |
| `feature/` | new notebook section or technique |
| `fix/` | bug fix |
| `data/` | data preparation or exploration |
| `docs/` | README, comments, documentation |
| `refactor/` | code reorganization in `src/` |

### Examples

```
feature/otsu-segmentation
fix/iou-calculation
data/explore-val-split
docs/update-readme
```

## Setup apos clonar

```bash
# 1. Criar e ativar o ambiente virtual
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt

# 2. Configurar o filtro git que limpa outputs dos notebooks
#    (remove paths locais, tokens e outputs antes de commitar)
git config filter.strip-notebook.clean '.venv/bin/python scripts/clean_notebook.py'
git config filter.strip-notebook.smudge cat
```

O filtro e obrigatorio para evitar que dados pessoais (paths, tokens Kaggle)
sejam commitados nos outputs das celulas do notebook.
