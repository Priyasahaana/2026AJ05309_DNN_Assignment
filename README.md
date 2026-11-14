# DNN Assignment 2 - RNN, LSTM, GRU and Transformer (2026AJ05309)

## RNN, LSTM, GRU and Transformer Models

**Student Name:** PRIYASAHAANA M.  
**Student ID:** 2026AJ05309

## Student variant
Student ID ends in 9 (odd): task 5(c) is the LSTM forget-gate bias ablation (bias 0 versus 1) at T = 200.

## Run time on a T4
Total: see the final cell timings of the last Restart-and-Run-All (Problem 1 and Problem 2 are each run once).

## 1. Project Overview

This assignment implements and evaluates recurrent neural networks and Transformer-based architectures on two tasks:

- **Problem 1 – Sequence Modelling:** Human Activity Recognition (HAR) classification using RNN, LSTM and GRU models, followed by the Adding Problem to study long-term dependency learning and gradient behaviour.
- **Problem 2 – Attention and Transformers:** From-scratch scaled dot-product attention, multi-head attention and Transformer encoder components, followed by HAR classification and character-level Shakespeare language modelling.

All neural network experiments use PyTorch. The custom recurrent cells and attention components are implemented from scratch for the required implementation exercises. No pretrained model weights are used.

## 2. Environment and Dependencies
- Python 3.10+ (developed on Colab, Tesla T4 GPU), PyTorch 2.x, NumPy, pandas, Matplotlib, scikit-learn.
- Exact versions: `requirements.txt` (`pip install -r requirements.txt`). seaborn is optional (heatmaps fall back to Matplotlib).
- No pre-trained weights. Datasets are downloaded by the notebooks into `data/` (UCI HAR, Tiny Shakespeare); the adding problem is generated in code.
- 
Experiments were conducted in Google Colab using the following reported environment:

| Component | Version |
|---|---|
| Python | 3.13.16 |
| PyTorch | 2.14.1+cu130 |
| NumPy | 2.4.6 |
| Pandas | 2.2.3 |
| Matplotlib | 3.10.0 |
| Scikit-learn | 1.6.1 |
| Seaborn | 0.13.2 |

Install the dependencies using:

```bash
pip install -r requirements.txt
```

A CUDA-enabled GPU was used for the computationally intensive experiments where available. Execution time may vary depending on hardware.

## 3. Project Structure

```text
2026AJ05309_DNN_Assignment/
│
├── README.md
├── requirements.txt
├── report.pdf
├── results.csv
├── detailed_results.csv
│
├── problem1/
│   ├── Problem1.ipynb
│   └── figures/
│
└── problem2/
    ├── Problem2.ipynb
    ├── generated_samples.txt
    ├── figures/
    └── models/
```

The notebooks generate experiment outputs, figures, metrics and model checkpoints in their respective problem folders. The consolidated `results.csv` contains one row per experiment run, while `detailed_results.csv` stores additional experiment-level measurements.

## 4. Execution Instructions

1. Open the required notebook in Google Colab or Jupyter Notebook.
2. Install the dependencies using the provided `requirements.txt`, if required.
3. Ensure that the required dataset or data-download source is accessible.
4. Execute the notebook cells sequentially from top to bottom.
5. Check the generated metrics, tables and figures.
6. Confirm that the output files are saved in the corresponding problem folder.

**Problem 1:** Run `problem1/Problem1.ipynb` to reproduce the recurrent-cell verification, HAR classification, Adding Problem experiments, gradient analysis and ablation studies.

**Problem 2:** Run `problem2/Problem2.ipynb` to reproduce the attention implementation checks, Transformer HAR experiments, Shakespeare language modelling, attention visualisations and ablation studies.

## 5. Experimental Configuration and Key Findings

Experiments used random seeds **13, 309 and 2004** where applicable.

### Problem 1

- Custom RNN, LSTM and GRU cells were checked against corresponding PyTorch implementations.
- On HAR classification, the GRU achieved the highest mean test accuracy (**89.63%**) and macro-F1 (**89.70%**) among the three recurrent models.
- The LSTM achieved mean test accuracy of **89.49%** and macro-F1 of **89.52%**.
- The vanilla RNN achieved mean test accuracy of **78.53%** and macro-F1 of **78.30%**.
- In the Adding Problem, GRU performed substantially better than RNN and LSTM at shorter sequence lengths. Performance deteriorated as sequence length increased, and none of the evaluated models solved the T=400 setting under the selected training configuration.
- Gradient-flow and ablation experiments were used to investigate the effect of sequence length, hidden size and LSTM forget-gate bias.

### Problem 2

- Scaled dot-product attention, multi-head attention, sinusoidal positional encoding and Transformer encoder components were implemented and tested.
- The Transformer achieved mean HAR test accuracy of **91.46%** and macro-F1 of **91.48%** across three seeds.
- The Shakespeare Transformer achieved a full sequential validation loss of **1.7596 nats**, equivalent to **2.5386 bits per character (BPC)**.
- Positional-encoding, attention-head and context-length experiments were conducted.
- A separate no-causal-mask experiment demonstrated future-token leakage. Its low validation loss is therefore not a valid autoregressive language-modelling result and is included only as an ablation demonstrating why causal masking is necessary.

## 6. Results and Reproducibility

- `results.csv`: Consolidated results with the required columns: problem, model, seed, sequence length, test accuracy, macro-F1, test MSE, parameter count, seconds per epoch, validation loss, bits per character and notes.
- `detailed_results.csv`: Additional experiment measurements, including validation metrics, parameter counts, timing and attention complexity.
- `problem1/figures/`: Problem 1 plots and visualisations.
- `problem2/figures/`: Problem 2 plots and visualisations.
- `problem2/generated_samples.txt`: Shakespeare text generated at temperatures 0.2, 0.8 and 1.5.
- `problem2/models/`: Saved model checkpoints used in the experiments.

The reported results correspond to the recorded experiment runs. Small differences may occur when experiments are rerun because of hardware, software and randomisation differences.

## 7. Limitations

The hidden-size sweep was conducted on the Adding Problem at T=200 rather than as a HAR hidden-size sweep. The no-mask language-modelling experiment is intentionally included to illustrate information leakage and must not be interpreted as a valid causal language model. Results are limited to the datasets, seeds, architectures and training configurations evaluated in this assignment.

## 8. References

1. PyTorch Documentation: https://pytorch.org/docs/stable/
2. UCI Machine Learning Repository – Human Activity Recognition Using Smartphones: https://archive.ics.uci.edu/dataset/240/human+activity+recognition+using+smartphones
3. The Tiny Shakespeare Dataset: https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt
4. Vaswani et al. (2017), *Attention Is All You Need*: https://arxiv.org/abs/1706.03762
