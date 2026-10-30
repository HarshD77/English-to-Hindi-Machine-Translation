# English-to-Hindi Machine Translation

## Problem statement and objective

Machine translation between English and Hindi is challenging because the languages differ in
word order, grammar, morphology, and writing script. This project investigates how different
neural language-model architectures perform on English-to-Hindi translation.

The objectives are to:

- prepare and evaluate a parallel English-Hindi dataset;
- implement an encoder-decoder Transformer from scratch;
- fine-tune a multilingual sequence-to-sequence model;
- compare decoder-only large language models using prompting and LoRA fine-tuning; and
- evaluate translation quality, inference cost, sampling temperature, and output reliability.

## Dataset

The project uses the [IIT Bombay English-Hindi parallel corpus](https://huggingface.co/datasets/cfilt/iitb-english-hindi)
from Hugging Face (`cfilt/iitb-english-hindi`). The data-preparation notebook cleans the
sentence pairs, removes unsuitable examples, creates reproducible splits, and generates
responsible-AI probe sentences.

The prepared dataset contains:

| Split | Sentence pairs |
|---|---:|
| Training | 50,000 |
| Validation | 500 |
| Test | 500 |

Prepared files are stored in `fgai_mt_en_hi/data/` as `train.jsonl`, `val.jsonl`, and
`test.jsonl`. Dataset statistics, length distributions, and probe sentences are stored in
the same directory.

## Models implemented

### Transformer from scratch

A small encoder-decoder Transformer is trained from scratch using a shared 16k SentencePiece
BPE tokenizer. This provides a baseline without pretrained language-model knowledge.

### mT5-small

`google/mt5-small` is a multilingual encoder-decoder model fine-tuned on the prepared
English-Hindi training data. It uses full model fine-tuning and a multilingual SentencePiece
tokenizer.

### LLaMA

`meta-llama/Llama-3.2-1B-Instruct` is a decoder-only instruction model evaluated first with
three-shot prompting and then adapted with LoRA. LoRA updates only a small number of
trainable parameters while keeping the base model mostly frozen.

### Gemma

`google/gemma-3-1b-it` is a decoder-only instruction model evaluated with the same prompting
and LoRA approach as LLaMA. This enables a direct comparison between two modern
decoder-only models.

## Execution steps

Run the notebooks in the following order:

1. `00_data_preparation.ipynb` - Download, clean, filter, split, and save the dataset.
2. `01_transformer_from_scratch.ipynb` - Train and evaluate the scratch Transformer.
3. `02_mt5_finetuning.ipynb` - Fine-tune and evaluate mT5-small.
4. `03_llama_translation.ipynb` - Run LLaMA prompting, LoRA fine-tuning, and evaluation.
5. `04_gemma_translation.ipynb` - Run Gemma prompting, LoRA fine-tuning, and evaluation.
6. `05_comparison_and_analysis.ipynb` - Compare all models and analyze temperature and
   responsible-AI probes.

The notebooks are designed for Google Colab but can also be run locally with Jupyter and a
compatible Python environment. GPU execution is recommended, especially for mT5, LLaMA, and
Gemma.

### Required dependencies

```bash
pip install datasets transformers sentencepiece sacrebleu pandas matplotlib \
    torch peft accelerate bitsandbytes jupyter
```

The LLaMA and Gemma notebooks require a Hugging Face token and accepted access to the
corresponding gated models:

```python
from huggingface_hub import login
login()
```

## Results

The main results are saved in `fgai_mt_en_hi/results/metrics_summary.csv`. BLEU and chrF
are reported on the held-out test set. Higher values indicate better translation quality.

| Model | Architecture | Adaptation | BLEU | chrF | Inference ms/sentence |
|---|---|---|---:|---:|---:|
| Transformer from scratch | Encoder-decoder | Trained from scratch | 3.97 | 24.39 | 18.5 |
| mT5-small | Encoder-decoder | Full fine-tuning | 4.13 | 23.23 | 115.6 |
| LLaMA-3.2-1B-Instruct | Decoder-only | LoRA fine-tuning | 10.46 | 35.02 | 212.4 |
| Gemma-3-1B-IT | Decoder-only | LoRA fine-tuning | 13.55 | 38.99 | 264.4 |

Additional outputs include:

- `translations_all_models.csv` - Test-set translations from all models
- `comparison_overview.png` - Visual model comparison
- `output_diagnostics.csv` - Hallucination and failure-proxy diagnostics
- `responsible_ai_probes.csv` - Outputs for probe sentences
- `temperature_bleu.csv` and `temperature_chrf.csv` - Sampling-temperature results

Results may vary with hardware, library versions, random seeds, and training settings.

## Model comparison and evaluation

The scratch Transformer is the fastest model, but its limited capacity and lack of
pretraining result in the lowest BLEU score. mT5 improves BLEU slightly, although its chrF
score is lower than the scratch Transformer in this experiment and full fine-tuning is more
expensive.

The decoder-only models perform substantially better after LoRA fine-tuning. LLaMA achieves
higher scores than both encoder-decoder baselines, while Gemma achieves the best BLEU and
chrF scores. The comparison also considers parameter count, trainable parameter count,
training time, inference latency, translation samples, temperature behavior, and diagnostic
outputs rather than relying on a single metric.

## Sampling-temperature analysis

Sampling temperature was evaluated at 0.0, 0.3, 0.7, 1.0, and 1.3. The best results for
every model were obtained at temperature 0.0 or 0.3:

| Temperature | Transformer BLEU | mT5 BLEU | LLaMA BLEU | Gemma BLEU |
|---:|---:|---:|---:|---:|
| 0.0 | 3.99 | 4.06 | 9.02 | 11.67 |
| 0.3 | 2.41 | 3.74 | 8.29 | 11.72 |
| 0.7 | 2.67 | 3.00 | 5.80 | 8.51 |
| 1.0 | 0.96 | 1.13 | 2.06 | 3.08 |
| 1.3 | 0.14 | 0.23 | 0.04 | 0.14 |

Increasing temperature generally reduces BLEU and chrF because it introduces more varied
and less deterministic outputs. For this translation task, greedy or near-greedy decoding
(temperature 0.0-0.3) is preferable to high-temperature sampling. The complete chrF
results are available in `temperature_chrf.csv`.

## Conclusion

`google/gemma-3-1b-it` with LoRA fine-tuning is the best-performing model in this project,
achieving **13.55 BLEU** and **38.99 chrF**. It outperforms the LLaMA LoRA model and both
encoder-decoder baselines on the held-out test set. The scratch Transformer remains the
fastest option, so model selection should balance translation quality against inference
latency and available compute. For the best quality, Gemma with low-temperature decoding
is the recommended configuration.

## Repository structure

```text
.
├── 00_data_preparation.ipynb
├── 01_transformer_from_scratch.ipynb
├── 02_mt5_finetuning.ipynb
├── 03_llama_translation.ipynb
├── 04_gemma_translation.ipynb
├── 05_comparison_and_analysis.ipynb
└── fgai_mt_en_hi/
    ├── data/
    └── results/
```

## License and data attribution

This repository contains course/project code and experiment outputs. Review the license and
usage conditions of the IIT Bombay dataset and each pretrained model before redistribution
or commercial use.
