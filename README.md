# ChromeTransformer

A Transformer-based model for binary gene expression classification (low/high) from histone modification signals, compared to DeepChrome and AttentiveChrome across 56 human cell types.

---

## 1. Project Overview

This project explores whether a Transformer encoder can learn to predict gene expression from histone modification data, and whether its attention mechanism offers more interpretations of how the model learns gene regulation patterns, rather than being a black box.

The model takes histone mark signals across 100 genomic bins for each gene and predicts whether that gene is highly expressed or silenced. It was trained and evaluated on 56 cell types from the REMC dataset, achieving a mean AUROC of 0.8044 and reaching 0.9178 on the best-performing cell type (E123, left ventricle).

The full project consists of three stages: replicating DeepChrome and AttentiveChrome as baselines, training ChromeTransformer, and interpreting what the model learned through attention map analysis.

---

## 2. Biological Motivation

Gene expression is controlled by the chromatin domain surrounding each gene. Histone modifications, which are chemical tags added to histone proteins, act as a regulatory code: some marks like H3K4me3 are associated with active promoters, others like H3K27me3 with silenced regions, and marks like H3K4me1 with distal enhancers that can activate genes from tens of thousands of base pairs away.

Predicting expression from these marks is not just a classification problem, it is a way to interpret that regulatory factor computationally. Two prior models have approached this task. DeepChrome (Singh et al., 2016) used a CNN to capture local chromatin patterns around the transcription start site. AttentiveChrome (Singh et al., 2019) extended this with a hierarchical LSTM and attention mechanism, allowing the model to weight different genomic regions differently. We have already replicated these two models [LINK to repo](https://github.com/Mariambadra/deepchrome).

Both models, however, are architecturally constrained in how they model long-range interactions. A CNN has a fixed receptive field. An LSTM processes bins sequentially and struggles to directly connect distant positions. The biological reality is that enhancers can be located far upstream or downstream of the genes they regulate, and a model that cannot directly relate distal bins to the promoter region may be missing a meaningful signal.

Transformers, by design, allow every position to attend to every other position simultaneously. This makes them a natural fit for the question: can a model learn that the H3K4me1 signal in a distal bin is relevant to H3K4me3 activity at the TSS?

---

## 3. Model Architecture

ChromeTransformer takes a (batch, 100, 5) tensor as input: 100 bins, 5 histone marks per bin, and outputs binary logits for high vs. low expression.

**Input projection.** A linear layer projects the 5 histone mark values at each bin into a higher-dimensional space (d_model), followed by LayerNorm and dropout. No bias is used in the projection.

**Positional embedding.** A learned embedding table of shape (100, d_model) is added to encode bin position. Unlike sinusoidal embeddings, learned embeddings let the model discover which positional relationships matter for this specific task.

**Encoder blocks.** N stacked encoder blocks, each following a Pre-LN design:
- LayerNorm → Multi-head self-attention (batch_first) → residual connection
- LayerNorm → Feed-forward network (d_model → d_model × ffn_mult → d_model, ReLU) → residual connection

**Classifier head.** After the final LayerNorm, the bin representations are mean-pooled across the 100 bins to produce a single gene-level vector, which is passed through a two-layer classifier (Linear → ReLU → Dropout → Linear) to produce raw logits.

**Loss.** Cross-entropy with class weights to handle label imbalance (~28.7% positive in the test set).

**Best hyperparameters** (tuned via Optuna on cell type E047, 30 trials):

| Parameter | Value |
|---|---|
| d_model | 128 |
| n_heads | 4 |
| n_layers | 3 |
| ffn_mult | 2 |
| dropout | 0.134 |
| learning rate | 0.0061 |
| batch size | 128 |
| weight decay | 9e-6 |

---

## 4. Results

Each model is trained separately per cell type. Metrics reported are AUROC and AUPR on the held-out test set.

| Model | Mean AUROC | Mean AUPR |
|---|---|---|
| DeepChrome (paper) | 0.8000 | — |
| AttentiveChrome (paper) | 0.8120 | — |
| ChromeTransformer (ours) | 0.8044 | 0.4916 |

**Top performing cell types:**

| Cell Type | AUROC | AUPR |
|---|---|---|
| E123 (Left ventricle) | 0.9178 | 0.857 |
| E117 (HeLa cells) | 0.9114 | 0.836 |
| E116 (GM12878) | 0.8995 | 0.832 |

**Lowest performing cell types:**

| Cell Type | AUROC |
|---|---|
| E112 | 0.7032 |
| E084 | 0.7251 |
| E096 | 0.7264 |

One important caveat: Optuna hyperparameter tuning was run on E047 only, and the resulting configuration was applied uniformly to all 56 cell types. Per-cell-type tuning would likely push the mean AUROC higher and close the gap with AttentiveChrome.

---

## 5. Key Findings

**Attention maps reflect real biology.** Across the three cell types we analyzed in depth (E123, E117, E112), the attention patterns were not random — they aligned with known chromatin features in interpretable ways.

**Layer 1 detects, Layer 2 integrates.** Layer 1 attention is consistently spiky and concentrated on specific bins, with peaks aligned to active histone marks (H3K4me3, H3K4me1) at the TSS. Layer 2 attention spreads broadly across the gene, consistent with the model building a global representation from the local features Layer 1 identified.

**Active and repressed genes are handled differently.** For high-expression genes, Layer 2 distributes attention across the full gene body — consistent with integrating enhancer signals from distal bins. For low-expression genes, Layer 2 anchors attention to the distal end of the gene (bins 0–8 in E123), where H3K27me3 repressive signal is elevated. This asymmetry is the clearest evidence that the model learned distinct regulatory strategies for active vs. silenced genes.

**Interpretability degrades gracefully with performance.** E123 (AUROC 0.9178) shows sharp, biologically anchored attention patterns. E112 (AUROC 0.7032) shows diffuse, competing attention foci with no clear anchor — reflecting the model's genuine confusion on a cell type where chromatin signals are harder to discriminate. The attention maps track model confidence in a meaningful way.

**The TSS is not the whole story.** In E117, Layer 2 attention develops a dip at the TSS for high-expression genes, redistributing to flanking regions. This is consistent with enhancer-promoter communication and is the kind of pattern a CNN or sequential LSTM would be unlikely to discover.

---

## 6. How to Run

### Requirements

```
torch
numpy
pandas
scikit-learn
optuna
matplotlib
```

### Data

Download the dataset from [Zenodo record 2652278](https://zenodo.org/record/2652278). The expected directory structure is:

```
data/
  {cell_type}/
    classification/
      train.csv
      valid.csv
      test.csv
```

Each CSV has no header and 8 columns: `GeneID | BinID | H3K27me3 | H3K36me3 | H3K4me1 | H3K4me3 | H3K9me3 | Label`

### Training

Open `chromtransformer_train.ipynb`. Set `DATA_ROOT` to your data directory and `WEIGHTS_ROOT` to where you want checkpoints saved. Run all cells. Optuna tuning runs on E047 first, then trains one model per cell type sequentially.

### Interpretation

Open `chromtransformer_interpret.ipynb`. Add your saved weights as a dataset input. Run the architecture definition cells first (ChromeDataset, ChromeTransformer), then run the interpretation cells. Figures are saved to `/kaggle/working/interpret_outputs/` or your configured output directory.

### Pretrained weights

Pretrained weights for all 56 cell types are available on Kaggle: `chromtransformer-weights`. Each `.pt` file contains the model state dict, hyperparameters, and test metrics for that cell type.

---

## Repository Structure

```
deepchrome/
  chrome_dataset.py        # Dataset class
  chrome_transformer.py    # Model architecture
  chrome_train.py          # Training loop and Optuna tuning
  chrome_interpret.py      # Attention extraction and visualization
  auroc_results.csv        # Per-cell-type AUROC and AUPR results
```

---

## References

- Singh, R. et al. (2016). DeepChrome: deep-learning for predicting gene expression from histone modifications. *Bioinformatics*.
- Singh, R. et al. (2019). AttentiveChrome: attention-based neural networks for predicting gene expression from histone modifications. *NeurIPS*.
- Roadmap Epigenomics Consortium (2015). Integrative analysis of 111 reference human epigenomes. *Nature*.
