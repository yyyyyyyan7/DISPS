# Grid Search Procedure & Sensitivity Analysis

> **FMR (FastMatchRate)**  
> Probability that a document is directly assigned to an existing topic during streaming **without** requiring a second round of processing. Higher FMR indicates stronger online stability and lower maintenance overhead.

## Procedure

We conducted a **two-stage grid search** to tune the parameters $\alpha$ and $\beta$, jointly considering three criteria:  
- **NMI**: clustering quality  
- **FMR**: online stability and fast assignment  
- **Runtime**: processing efficiency

All experiments were carried out using the **GTE-DISPS** framework; **consistent trends** were also observed for **BGE-DISPS** and **GIST-DISPS**.

To balance these criteria, we define a composite score:

$$
\mathrm{Score} = \mathrm{NMI}\cdot \mathrm{FMR} \cdot \bigl(1-\mathrm{NormTime}\bigr)
$$

where larger values indicate **better clustering accuracy**, **stronger online stability**, and **lower latency**.  
Here, $\mathrm{NormTime}\in[0,1]$ denotes the runtime normalized via min–max scaling per dataset under the same experimental setting.

---

## Coarse Grid Search

We first explored a wide range of $(\alpha, \beta)$ to understand overall trends.  
Representative results on **TSet Dataset** are shown below.

- **NMI** remains *consistently high* across most $(\alpha,\beta)$ settings, indicating **low sensitivity** to these parameters. The best NMI concentrates around **moderate-to-high $\alpha$** with **small-to-moderate $\beta$**, with several nearby configurations showing comparably strong quality.  
- In contrast, **FMR** and **Time** vary more substantially. Increasing **$\beta$** generally **raises FMR** and **reduces Time**, but **too large $\beta$** degrades **NMI**. The influence of **$\alpha$** on efficiency is **secondary** and can be **non-monotonic**; at high $\beta$, larger $\alpha$ can further suppress NMI.  
- Guided by these observations, we refine the search to **$\alpha \in [0.60, 0.80]$** and **$\beta \in [0.70, 0.80]$** to balance accuracy and efficiency.

### Representative Coarse Grid (TSet Dataset)

> **FMR in %**, **Time in seconds**.

| α | β | NMI | FMR (%) | Time (s) |
|---:|---:|---:|---:|---:|
| 0.40 | 0.60 | 0.8694 | 53.82 | 1447.36 |
| 0.40 | 0.70 | 0.8734 | 63.69 | 1071.06 |
| 0.40 | 0.80 | 0.8434 | 76.12 | 625.03 |
| 0.50 | 0.60 | 0.8737 | 51.67 | 261.41 |
| 0.50 | 0.70 | 0.8790 | 62.97 | 260.02 |
| 0.50 | 0.80 | 0.8562 | 72.70 | 136.18 |
| 0.60 | 0.50 | 0.8748 | 39.05 | 79.37 |
| 0.60 | 0.60 | 0.8720 | 44.59 | 60.08 |
| 0.60 | 0.70 | 0.8720 | 61.52 | 29.42 |
| 0.60 | 0.80 | 0.8670 | 68.11 | 43.40 |
| 0.60 | 0.90 | 0.7994 | 80.93 | 23.30 |
| 0.70 | 0.70 | 0.8667 | 55.70 | 23.97 |
| 0.70 | 0.80 | 0.8616 | 73.60 | 18.31 |
| 0.70 | 0.90 | 0.7942 | 85.71 | 17.12 |
| 0.80 | 0.50 | 0.8992 | 38.91 | 47.05 |
| 0.80 | 0.60 | 0.8668 | 41.04 | 31.29 |
| 0.80 | 0.70 | 0.8489 | 55.42 | 22.04 |
| 0.80 | 0.80 | 0.8294 | 72.70 | 16.61 |
| 0.80 | 0.90 | 0.7852 | 78.41 | 15.38 |
| 0.90 | 0.50 | 0.8932 | 34.61 | 46.47 |
| 0.90 | 0.60 | 0.8919 | 39.06 | 37.45 |
| 0.90 | 0.70 | 0.8730 | 52.12 | 26.37 |
| 0.90 | 0.80 | 0.8133 | 69.05 | 18.24 |
| 0.90 | 0.90 | 0.7459 | 83.53 | 15.79 |

---

## Refined Grid Search

Building on the coarse search, we conducted a finer exploration within  
**$0.60 \le \alpha \le 0.80$** and **$0.60 \le \beta \le 0.80$**.  
The table below reports representative results evaluated by the **composite score**.

The configuration **$(\alpha=0.66,\ \beta=0.76)$** lies in a **performance plateau**, achieving **high NMI**, **strong FMR**, and **competitive runtime**.  
Nearby settings (e.g., $\alpha \in [0.64, 0.68]$, $\beta \in [0.74, 0.78]$) deliver **comparable results**, suggesting **low sensitivity** to small parameter variations within this region.

> **FMR in %**, **Time in seconds**.

| α | β | NMI | FMR (%) | Time (s) | Score |
|---:|---:|---:|---:|---:|---:|
| 0.62 | 0.74 | 0.873 | 81.2 | 21.5 | 0.579 |
| 0.64 | 0.76 | 0.878 | 82.5 | 20.9 | 0.597 |
| **0.66** | **0.76** | **0.881** | **83.1** | **21.1** | **0.607** |
| 0.68 | 0.76 | 0.879 | 82.7 | 22.0 | 0.593 |
| 0.70 | 0.78 | 0.877 | 81.6 | 22.4 | 0.580 |

---

## Sensitivity Analysis & Final Choice

The refined search reveals that optimal performance consistently arises around **$(\alpha=0.66,\ \beta=0.76)$**,  
with neighboring configurations yielding **similar accuracy, stability, and efficiency**.  
This **robustness across datasets** justifies adopting **$(\alpha=0.66,\ \beta=0.76)$** as the **default** configuration in the main experiments.

---

> **Notes for Reproducibility**  
> - `NormTime` is computed via **min–max** normalization within each dataset under the same experimental setting.  
> - All runs were performed under the **GTE-DISPS** backbone unless otherwise noted; **BGE-DISPS** and **GIST-DISPS** exhibit consistent trends.
