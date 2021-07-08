TECH-RANK algorithm
=======

## Scope
TechRank aims to help decision-makers to quantitatively assess the influence of the entities in order to  take good investments decisions under high level of uncertainty. 
Please refer to [the short paper](../docs/_static/TechRank_shortpaper.pdf)

------
### Code structure and usage
The code is a mix of notebooks and `py` files: 
- the _py files_ contain the functions needed in the notebook and the declaration of the classes;
- the _notebooks_ explain all the steps.

**Classes** 

We work with 3 dataclasses: `Companies`, `Technologies` and `Investors`.
In `main.ipynb`, we create the Companies and Technologies objects, while in `investors.ipynb` the Investors instances. All objects are saved as [pickle files](https://docs.python.org/3/library/pickle.html) in the `classes` folder. 
Please note that the Companies object changes in `investments.ipynb`, so this notebook must be run after `main.ipynb`.

**`main.ipynb`**
Table of contents:




------
## Hints of bliography:
The main *sources* of this work are the following:
- "The Building Blocks of Economic Complexity" by César A. Hidalgo and Ricardo Hausmann
https://rb.gy/rhrsi2
- "The Virtuous Circle of Wikipedia: Recursive Measures of Collaboration Structures" by Maximilian Klein. Thomas Maillart and John Chuang
https://dl.acm.org/doi/10.1145/2675133.2675286 \
code: https://github.com/wazaahhh/wiki_econ_capability

Please find the complete list on the bibliography of [the short paper](../docs/_static/TechRank_shortpaper.pdf) . 

