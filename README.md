# LTX-2.5 on Trainium 3

Generates video with audio with [LTX-2.5](https://huggingface.co/Lightricks/LTX-2.5-Diffusers)
on a trn3.48xlarge, its transformer split over the NeuronCores.

## Setup

The demo keeps its venv, the weights and the compiled graphs in a working
directory of its own, here `/data/video-gen-demo`. The weights are in
a gated repository: accept its terms at
[Lightricks/LTX-2.5-Diffusers](https://huggingface.co/Lightricks/LTX-2.5-Diffusers)
with the account of your Hugging Face token. Then, on a pod started from the
AWS Neuron image, [`setup.sh`](setup.sh) creates the working directory with
the venv and the weights (82 GB) in it, and writes the path of the directory
to `.env`:

```bash
./setup.sh /data/video-gen-demo
```

## Generate

[`run.sh`](run.sh) runs a script in the venv, with the variables of `.env`:

```bash
./run.sh generate.py              # 10 s at 1024x512, transformer and video decoder on NeuronCores 0 to 3, the rest on CPU
./run.sh generate.py --two-stage  # 10 s at 2048x1024, transformer and video decoder on NeuronCores 0 to 15
./run.sh generate.py --cpu        # all on CPU, to compare
```

With `--two-stage`, [`generate.py`](generate.py) follows the two-stage recipe of
the model card: 8 steps at 1024x512, the x2 latent upsampler on CPU, then 3
steps at 2048x1024.

[`generate.py`](generate.py) writes the clip to `outputs/`: an mp4 with its
audio, 4 of its frames as a png, and its frames and audio as an npz. `--prompt`
and `--seed` set what it generates.

## Checks

| Command | Checks |
|---------|--------|
| `./run.sh check_tp.py` | On CPU, that the transformer split over 2 ranks by [`parallel.py`](parallel.py) computes what it computes whole. |
| `./run.sh check_sp.py` | On CPU, that the transformer split over 2 × 2 ranks by [`parallel.py`](parallel.py), its weights by tensor parallelism and its video tokens by sequence parallelism, computes what it computes whole. |
| `./run.sh check_frame_split.py` | On CPU, that the video VAE split by frames over 4 ranks by [`parallel.py`](parallel.py) decodes what it decodes whole. |
| `./run.sh check_neuron.py` | On NeuronCores 0 and 1, that the transformer split over them computes what it computes on CPU, within the tolerance of the model support sweep. It takes the cores: run it when nothing else uses them. |
