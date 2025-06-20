# DISPS: Dual-Index Semantic Publish/Subscribe System

> Official implementation of our paper:  
> **"Semantic Publish/Subscribe over Evolving Topics"**  
> (*Under review*)  

DISPS is a scalable and low-latency publish/subscribe framework designed for real-time text streams. It organizes documents and subscriptions into evolving semantic topics through a dual-index architecture, leveraging dynamic pivot updates and the LocalPair Distance Metric for robust clustering and efficient matching.

---

## 🌐 Overview

Traditional keyword-based pub/sub systems fail to capture deep semantics and cannot handle topic evolution in real-time. DISPS addresses these limitations by introducing:

- **ProtoIndex**: A fast proto-clustering index that absorbs incoming documents into local groups.
- **TopicIndex**: A high-level semantic index that manages stable, evolving topics via pivot-based representations.
- **LocalPair Distance Metric**: A novel distance metric that enables robust cluster assignment and noise resistance.
- **Dynamic Matching**: Top-$k$ topic matching with incremental updates for subscription accuracy under drift.

---

## 📌 Features

- ⚡ **High throughput** on million-scale streaming data
- 🔁 **Dynamic topic evolution** with pivot aging and drift detection
- 🔍 **Model-agnostic design** supporting BGE, GIST, and GTE embeddings
- 🔧 **Plug-and-play interface** for real-time pub/sub integration
- 📊 Comprehensive benchmarking on short and long text datasets

---


