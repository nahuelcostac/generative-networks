# Generative Networks — Course Notebooks

Code accompanying the [Generative Networks course](https://www.nahuelcosta.com/teaching/generative-networks). Each notebook implements one chapter's ideas from scratch, small enough to train in minutes on a single GPU — or on Colab.

## Notebooks

| Chapter | Notebook | |
|---|---|---|
| 3. GANs | [gan/gan.ipynb](gan/gan.ipynb) — generator/discriminator on MNIST | [![Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/nahuelcosta/generative-networks/blob/main/gan/gan.ipynb) |
| 4. VAEs | [vae/vae.ipynb](vae/vae.ipynb) — ELBO, reparameterization | [![Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/nahuelcosta/generative-networks/blob/main/vae/vae.ipynb) |
| 4. VAEs | [vae/hierarchical_vae.ipynb](vae/hierarchical_vae.ipynb) — two latent levels | [![Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/nahuelcosta/generative-networks/blob/main/vae/hierarchical_vae.ipynb) |
| 5. Flows | [flows/normalizing_flow.ipynb](flows/normalizing_flow.ipynb) — invertible maps, exact likelihood | [![Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/nahuelcosta/generative-networks/blob/main/flows/normalizing_flow.ipynb) |
| 5. Flows | [flows/flow_matching.ipynb](flows/flow_matching.ipynb) — learned velocity field | [![Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/nahuelcosta/generative-networks/blob/main/flows/flow_matching.ipynb) |
| 6. Diffusion | [diffusion/diffusion.ipynb](diffusion/diffusion.ipynb) — DDPM forward/reverse process | [![Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/nahuelcosta/generative-networks/blob/main/diffusion/diffusion.ipynb) |
| 7. Transformers | [transformers/fundamentals.ipynb](transformers/fundamentals.ipynb) — BPE, attention, masks, positional encodings | [![Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/nahuelcosta/generative-networks/blob/main/transformers/fundamentals.ipynb) |
| 7. Transformers | [transformers/language_model.ipynb](transformers/language_model.ipynb) — decoder-only LM on Shakespeare | [![Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/nahuelcosta/generative-networks/blob/main/transformers/language_model.ipynb) |
| 8. Foundation models | [pretraining/mae.ipynb](pretraining/mae.ipynb) — masked autoencoders on CIFAR-10 | [![Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/nahuelcosta/generative-networks/blob/main/pretraining/mae.ipynb) |
| 8. Foundation models | [pretraining/contrastive_learning.ipynb](pretraining/contrastive_learning.ipynb) — SimCLR | [![Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/nahuelcosta/generative-networks/blob/main/pretraining/contrastive_learning.ipynb) |
| 8. Foundation models | [pretraining/cpc.ipynb](pretraining/cpc.ipynb) — contrastive predictive coding | [![Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/nahuelcosta/generative-networks/blob/main/pretraining/cpc.ipynb) |
| 8. Foundation models | [pretraining/lejepa.ipynb](pretraining/lejepa.ipynb) — Latent-Euclidean JEPA and SIGReg | [![Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/nahuelcosta/generative-networks/blob/main/pretraining/lejepa.ipynb) |
| 8. Foundation models | [pretraining/huggingface_sentiment.ipynb](pretraining/huggingface_sentiment.ipynb) — fine-tuning DistilBERT | [![Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/nahuelcosta/generative-networks/blob/main/pretraining/huggingface_sentiment.ipynb) |
| 9. Efficiency | [reduction/knowledge_distillation.ipynb](reduction/knowledge_distillation.ipynb) — SmolLM2 360M → 135M | [![Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/nahuelcosta/generative-networks/blob/main/reduction/knowledge_distillation.ipynb) |
| 9. Efficiency | [reduction/lora.ipynb](reduction/lora.ipynb) — LoRA and QLoRA adapters | [![Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/nahuelcosta/generative-networks/blob/main/reduction/lora.ipynb) |
| 9. Efficiency | [reduction/quantization.ipynb](reduction/quantization.ipynb) — float16, LLM.int8, NF4 | [![Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/nahuelcosta/generative-networks/blob/main/reduction/quantization.ipynb) |

Chapters 1 and 2 (introduction, probabilistic machine learning) are theory only.

## Running the notebooks

Each notebook is self-contained: it installs any missing package and downloads its own data on the first run. Nothing else in the repo is needed.

- **Colab** — click the badge and select a GPU runtime.
- **Locally** — Python 3.10+ with [PyTorch](https://pytorch.org/get-started/locally/) installed for your CUDA version, then `jupyter notebook`.

The generative notebooks (GAN, VAE, flows, diffusion) train on MNIST and run on CPU, slowly. The chapter 8 and 9 notebooks fine-tune real pretrained models and expect a GPU; the quantization notebook additionally requires CUDA for the int8/NF4 paths.

## Contributions welcome

This material is meant to keep improving. If you spot an error, something that no longer runs, or an explanation that could be clearer, please [open an issue](https://github.com/nahuelcosta/generative-networks/issues). Pull requests with fixes, new experiments, or extensions to a chapter are very welcome as well!
