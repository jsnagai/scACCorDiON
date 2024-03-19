
Welcome to scACCorDiON's documentation!
=====================================

scACCorDiON use the single-cell RNA sequencing patient cohort data to perform sample(patient) clustering. Our approach is done in two main steps: 1) Using Optimal Transport we learn a graph-to-graph metric and 2) A clustering approach using the Wasserstein Barycenters are done to cluster the patients accordingly to the cell-cell communication strengh.

To install scACCorDiON:
```
pip install git+https://github.com/jsnagai/scACCorDiON/
```

```{toctree}
---
caption: Synthetic Data - Learning a Metric from Directed Weighted Graphs
maxdepth: 2
---
Global_Local_Comparison
```

```{toctree}
---
caption: Real World Data Benchmark - Learning a Metric from Directed Weighted Graphs
maxdepth: 2
---
SingleCellDemo
```

```{toctree}
---
caption: Full Tutorial of PDAC data
maxdepth: 3
---
PDAC_FullTutorial
```
