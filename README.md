# LTX-2.5 on Trainium 3

Generates video with audio with [LTX-2.5](https://huggingface.co/Lightricks/LTX-2.5-Diffusers)
on a trn3.48xlarge, its transformer split over the NeuronCores.

## Setup

On a pod started from the AWS Neuron image:

```bash
./setup.sh
source .venv/bin/activate
```

The weights are in a gated repository: accept its terms at
[Lightricks/LTX-2.5-Diffusers](https://huggingface.co/Lightricks/LTX-2.5-Diffusers)
with the account of your Hugging Face token. The excludes skip what the
pipeline does not load (82 GB are left):

```bash
hf download Lightricks/LTX-2.5-Diffusers --local-dir models/LTX-2.5-Diffusers \
    --exclude "transformer_full/*" \
    --exclude "ltx-2.5-22b-distilled-lora-450-bf16.safetensors" \
    --exclude "transformer/*-of-00008.safetensors" \
    --exclude "connectors/diffusion_pytorch_model.safetensors"
```

## Generate

```bash
python generate.py         # transformer on NeuronCores 0 and 1, the other models on CPU
python generate.py --cpu   # all on CPU, to compare
```

[`generate.py`](generate.py) writes the clip to `outputs/`: an mp4 with its audio,
4 of its frames as a png, and its frames and audio as an npz. `--prompt` and
`--seed` set what it generates.

## Checks

| Command | Checks |
|---------|--------|
| `python check_tp.py` | On CPU, that the transformer split over 2 ranks by [`parallel.py`](parallel.py) computes what it computes whole. |
| `python check_neuron.py` | On NeuronCores 0 and 1, that the transformer split over them computes what it computes on CPU, within the tolerance of the model support sweep. It takes the cores: run it when nothing else uses them. |
