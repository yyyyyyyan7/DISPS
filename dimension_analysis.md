# Embedding Dimension Analysis

This document provides additional experimental results to address **Reviewer Comment D11**, which asked for a detailed analysis of embedding dimensionality.  
We systematically compared **384, 768, 1024, and 1536 dimensions** on the **T dataset**, and evaluated the trade-off between clustering quality, online matching stability, and computational efficiency.

---

## Evaluation Metric

To jointly evaluate these aspects, we define a **composite score function**:

$$
\text{Score} = \text{NMI} \times \text{FMR} \times (1 - \text{NormTime})
$$


where:  
- **NMI**: Normalized Mutual Information with ground-truth topics (higher is better).  
- **FMR (FastMatchRate)**: The probability that a document is directly assigned to an existing topic *during streaming* without requiring reprocessing (higher means more stable online clustering).  
- **NormTime**: Processing time normalized into the range \([0,1]\) (lower is better, so we use \(1-\text{NormTime}\)).  

This formulation ensures that embedding dimensionality is evaluated not only by clustering accuracy but also by its real-time efficiency and streaming robustness.

---

## Experimental Results

The following figure summarizes the results:

![Embedding Dimension Comparison](./figs/embedding_score_vs_nmi.png)

- **384 / 768 dimensions**:  
  Semantic information is overly compressed. Topic boundaries become blurred, and the FastMatchRate drops significantly, indicating that many documents require reprocessing instead of being matched directly.  

- **1536 dimensions**:  
  While semantic expressiveness increases slightly, the curse of dimensionality reduces distance discriminability. This leads to unstable clustering behaviors, fragmented topics, and increased runtime (larger NormTime).  

- **1024 dimensions**:  
  Strikes the best balance. It preserves sufficient semantic granularity to achieve higher **NMI**, maintains a consistently strong **FMR**, and avoids excessive computational cost.  

---

## Key Observations

1. **Trade-off Balance**  
   - Low-dimensional embeddings (384/768) are efficient but sacrifice semantic resolution.  
   - High-dimensional embeddings (1536) provide richer features but introduce instability and overhead.  
   - **1024 dimensions consistently outperform other choices in overall score.**

2. **FastMatchRate (FMR) as a Critical Factor**  
   - FMR reflects the *streaming nature* of our system: a higher FMR means fewer documents need secondary processing, directly boosting system throughput.  
   - This metric differentiates our analysis from conventional offline clustering, emphasizing the practical impact of dimensionality in **real-time publish/subscribe**.

3. **Final Decision**  
   - Based on the above, we adopt **1024 dimensions** as the default setting in DISPS.  
   - This choice is not arbitrary, but empirically validated by experiments balancing **accuracy (NMI)**, **robustness (FMR)**, and **efficiency (NormTime)**.

---

## Conclusion

These experiments validate that **1024-dimensional embeddings** provide the most favorable trade-off between semantic expressiveness and runtime efficiency in our framework.  
Detailed results and the evaluation formula are made available here to ensure transparency and reproducibility of our study.
