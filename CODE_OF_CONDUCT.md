# Develop Guidelines
## environment prepare
```
conda create -n samar python=3.9
conda activate samar
pip install poetry
poetry install
```
## pre commit
```
poe format
poe test
```
## release and pubish
```
git tag vX.X.X
git push origin vX.X.X
```