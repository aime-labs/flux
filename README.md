# FLUX
by Black Forest Labs: https://blackforestlabs.ai

![grid](assets/grid.jpg)

This repo is intended to run FLUX text-to-image and image-to-image models as worker for the scalable [AIME API Server](https://github.com/aime-team/aime-api-server).

## Setup & Installation

### Method1: Installation with Python venv

```bash
cd $HOME && git clone https://github.com/black-forest-labs/flux
cd $HOME/flux
python3.10 -m venv .venv
source .venv/bin/activate
pip install -e ".[all]"
```

### Method2: Installation with AIME MLC

Use [AIME MLC](https://github.com/aime-team/aime-ml-containers) to install in a dockerized container.

```bash
mlc-create flux1dev Pytorch 2.4.0
mlc-open flux1dev

sudo apt install libgl1-mesa-glx libglib2.0-0
pip install -r requirements.txt
pip install -e .
```

### Models

These models are avaible:

- `FLUX.1 [dev]` guidance-distilled variant
- `FLUX.1 [schnell]` guidance and step-distilled variant

| Name               | HuggingFace repo                                        | License                                                               | md5sum                           |
| ------------------ | ------------------------------------------------------- | --------------------------------------------------------------------- | -------------------------------- |
| `FLUX.1 [schnell]` | https://huggingface.co/black-forest-labs/FLUX.1-schnell | [apache-2.0](model_licenses/LICENSE-FLUX1-schnell)                    | a9e1e277b9b16add186f38e3f5a34044 |
| `FLUX.1 [dev]`     | https://huggingface.co/black-forest-labs/FLUX.1-dev     | [FLUX.1-dev Non-Commercial License](model_licenses/LICENSE-FLUX1-dev) | a6bd8c16dfc23db6aee2f63a2eba78c0 |
                         |

The weights of the autoencoder are also released under [apache-2.0](https://huggingface.co/datasets/choosealicense/licenses/blob/main/markdown/apache-2.0.md) and can be found in either of the two HuggingFace repos above. They are the same for both models.

## Command Line Usage

The weights will be downloaded automatically from HuggingFace once you start one of the demos. To download `FLUX.1 [dev]`, you will need to be logged in, see [here](https://huggingface.co/docs/huggingface_hub/guides/cli#huggingface-cli-login).
If you have downloaded the model weights manually, you can specify the downloaded paths via environment-variables:

```bash
export FLUX_SCHNELL=<path_to_flux_schnell_sft_file>
export FLUX_DEV=<path_to_flux_dev_sft_file>
export AE=<path_to_ae_sft_file>
```

For interactive sampling run

```bash
python -m flux --name <name> --loop
```

Or to generate a single sample run

```bash
python -m flux --name <name> \
  --height <height> --width <width> \
  --prompt "<prompt>"
```

## Start as AIME API Worker

Run following command line:

```bash
python3 main.py --ckpt_dir {{directory where the model was downloaded}} --api_server https://{{ url to your api server}}
```

This will register the worker to an [AIME API Server](https://github.com/aime-team/aime-api-server) to process distributed API job requests.
