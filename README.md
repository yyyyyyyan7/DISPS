# DISPS: Dual-Index Semantic Publish/Subscribe System

> Official implementation of our paper:  
> **"Semantic Publish/Subscribe over Evolving Topics"**  
> *(Under review)*

DISPS is an advanced, scalable publish/subscribe framework tailored for real-time semantic-aware text stream processing. It innovatively combines dynamic topic modeling with efficient approximate search, enabling precise and low-latency dissemination of textual information in fast-evolving environments.

---

##  Overview

Conventional publish/subscribe (pub/sub) systems predominantly rely on keyword or syntactic matching, which often leads to imprecise content delivery, limited semantic understanding, and inability to adapt to topic drift over time. These limitations pose significant challenges in scenarios such as social media monitoring, news aggregation, and intelligent notification services, where semantics and evolving user interests are critical.

To overcome these challenges, DISPS introduces a **dual-index architecture** that models evolving semantic topics via dynamically maintained pivots and incorporates robust distance function to ensure accurate clustering and subscription matching.

**Key components:**

- **ProtoIndex**  
  A lightweight, low-latency proto-clustering index that quickly aggregates incoming documents into preliminary semantic groups. It enables rapid assimilation of large-scale streaming data by exploiting approximate nearest neighbor search, providing a solid foundation for topic formation.

- **TopicIndex**  
  A hierarchical semantic index managing stable and evolving topics represented by dynamically generated pivot embeddings. It supports topic evolution by continuously updating pivot positions, merging semantically close topics, and pruning obsolete ones, thereby maintaining a concise yet expressive topic space.

- **LocalPair Distance**  
  A novel distance function leveraging pairs of nearest points for enhanced cluster assignment accuracy. This function improves noise robustness and effectively balances representational fidelity with computational efficiency, outperforming conventional single-pivot or centroid-based measures.

- **Dynamic Top-$k$ Matching**  
  An incremental subscription matching mechanism that efficiently maintains top-$k$ relevant topics for each subscriber. It handles semantic drift in real-time, ensuring subscription results remain accurate and up-to-date as topics evolve.

---

##  Features & Advantages

- **High Throughput and Scalability**  
  Designed to process millions of documents in real-time, DISPS leverages approximate search structures and localized computations to achieve low-latency clustering and subscription matching, making it suitable for large-scale production environments.

- **Adaptive Topic Evolution**  
  Incorporates mechanisms for pivot aging, dynamic pivot generation, and topic merging, allowing topics to evolve naturally with incoming data streams, capturing emerging trends and fading interests without manual intervention.

- **Robust Semantic Matching**  
  Uses advanced embedding models (BGE, GIST, GTE) in a model-agnostic manner, enabling precise semantic understanding and matching beyond lexical overlaps.

- **Flexible and Modular Architecture**  
  The dual-index design supports seamless integration with diverse embedding generators and downstream pub/sub applications, facilitating extensibility and customization.

- **Comprehensive Experimental Validation**  
  Validated on multiple benchmark datasets encompassing both short-text (tweets, social media posts) and long-text (news articles, scientific abstracts) streams, demonstrating superior clustering quality, subscription accuracy, and processing efficiency compared to state-of-the-art baselines.

- **Open Source and Ready for Deployment**  
  Provides production-quality code with detailed documentation, facilitating rapid adoption in research and industry projects focused on real-time semantic content distribution.

---

g++ -mavx main.cpp
