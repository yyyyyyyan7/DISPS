# Embedding Dimension Analysis

The goal of this analysis is to understand how different vector dimensions influence the balance between **clustering accuracy**, **online stability**, and **processing efficiency**.  
Such a study is crucial because the choice of dimensionality directly affects both **semantic expressiveness** and **system scalability** in real-time publish/subscribe scenarios.

We systematically compared **384, 768, 1024, and 1536 dimensions** on the **T dataset**.  
This dataset was chosen because it is representative of short-text streams, where semantic boundaries are often subtle and dimensionality plays a significant role.

---

## Evaluation Metric

To jointly evaluate clustering quality, online matching stability, and runtime efficiency, we define a **composite score function**:

$$
\text{Score} = \text{NMI} \times \text{FMR} \times (1 - \text{NormTime})
$$

where:  

- **NMI** (Normalized Mutual Information): Measures agreement with ground-truth topics. A higher value indicates better clustering accuracy.  
- **FMR (FastMatchRate)**: The probability that a document is directly assigned to an existing topic *during streaming*, without needing a second round of processing. A higher FMR reflects stronger online stability and reduced overhead.  
- **NormTime**: Processing time normalized to the range \([0,1]\). Lower values indicate better efficiency; we use \(1 - \text{NormTime}\) so that faster systems contribute positively to the score.  

This formulation ensures that no single factor dominates. Instead, it captures the **three key dimensions** that matter in our streaming system:  
1. **Accuracy (NMI)**: Do embeddings help separate topics correctly?  
2. **Stability (FMR)**: Do embeddings allow real-time assignment without reprocessing?  
3. **Efficiency (NormTime)**: Do embeddings support low-latency operation at scale?

---

## Experimental Results

To ensure **fairness and rigor**, we considered that different embedding dimensionalities may require different hyperparameter settings.  
Therefore, for each dimension (384, 768, 1024, 1536), we performed a **grid search** over clustering thresholds and related parameters, and report the **best-performing configuration** in the results below.  
This guarantees that the comparison reflects the intrinsic suitability of each dimensionality, rather than artifacts of suboptimal parameter choices.

The results are summarized in the following figure:

<p align="center">
  <img src="./figs/embedding_score_vs_nmi.png" alt="Embedding Dimension Comparison" width="550"/>
</p>

*Figure: Composite score vs. NMI under different embedding dimensions.  
Each point corresponds to the best configuration obtained via grid search for that dimensionality.*  

### Observations

- **384 / 768 dimensions**:  
  These embeddings compress semantic information too aggressively. As a result, topic boundaries become blurred and NMI drops. The **FMR** also decreases significantly, meaning that many documents fail to find their topic on the first attempt and require reprocessing.  

- **1536 dimensions**:  
  While higher-dimensional vectors provide slightly richer representations, they also suffer from the **curse of dimensionality**. Distance metrics become less discriminative, leading to unstable topic assignments and fragmented clusters. Moreover, computational cost increases substantially, reflected in larger normalized runtime values.  

- **1024 dimensions**:  
  This setting strikes the **best trade-off**. It preserves enough semantic granularity to achieve high **NMI**, maintains consistently strong **FMR**, and avoids the overhead associated with very high-dimensional vectors. The results demonstrate that 1024 dimensions deliver both stable clustering and efficient runtime performance.

---

## Key Insights

1. **Balance Between Quality and Efficiency**  
   Lower dimensions (384/768) run faster but compromise semantic resolution. Higher dimensions (1536) enrich semantics but introduce instability and inefficiency. **1024 dimensions consistently outperform all other choices** by balancing these two extremes.  

2. **The Role of FastMatchRate (FMR)**  
   Unlike conventional clustering metrics, **FMR** directly measures *streaming robustness*. A higher FMR means documents are more likely to be matched on the fly, reducing computational overhead. This metric highlights the **practical value** of embedding dimensionality in online systems, beyond static clustering accuracy.  

3. **Rigor Through Grid Search**  
   By performing grid search for **each embedding dimension**, we ensure that every comparison is based on its **best configuration**. This eliminates bias from parameter selection and makes the results more trustworthy.  

4. **Justification of Default Choice**  
   Based on empirical evidence, we adopt **1024 dimensions** as the default configuration in DISPS. This choice is not arbitrary but supported by systematic evaluation across accuracy (NMI), stability (FMR), and efficiency (NormTime).

---

## Conclusion

This analysis demonstrates that embedding dimensionality has a **non-trivial impact** on both the quality and scalability of real-time publish/subscribe systems.  
Our results confirm that **1024-dimensional embeddings**, when properly tuned, provide the most favorable balance, making them the best choice for DISPS.  

By sharing these results, we aim to increase transparency and reproducibility of our study. The **score function**, **grid search process**, and **raw experimental results** are available in this repository for further inspection.
