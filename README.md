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

[`generate.py`](generate.py) writes the clip to `outputs/`, named after the
demo and its size, such as `video-gen-demo-2048x1024`: an mp4 with its audio, 4
of its frames as a png, and its frames and audio as an npz. `--prompt` and
`--seed` set what it generates. With several prompts, it makes one clip per
prompt, one after the other, without loading the models again: their names end
with `-1`, `-2`...

## Films

With `--film`, [`generate.py`](generate.py) makes the shots of a film in the
same process, then the film, its shots edited one after the other:

```bash
./run.sh generate.py --two-stage --film films/fox.json  # outputs/fox-2048x1024-full.mp4, and its shots fox-2048x1024-shot-1.mp4...
```

A film is a JSON list of shots, each a prompt. A shot can start from the last
frames of an earlier shot, which it keeps as its first frames:

```json
{"prompt": "...", "from": {"shot": 2, "frames": 1}}
```

- From the last frame of a shot, a shot returns to its subject after a cut,
  with the same look, as prompts alone do not keep it.
- From the last 25 frames (1 s), a shot continues the motion of the shot.

A shot can also end on the first frames of a later shot, which the pipeline
takes as keyframes, so that the shot leads into it. The process then makes the
later shot first:

```json
{"prompt": "...", "to": {"shot": 3, "frames": 25}}
```

The number of frames is one more than a multiple of 8, as the VAE encodes the
frames after the first by 8. The film leaves out the frames a shot starts
from, which repeat the earlier shot, and the frames of a shot from its
keyframes on, which the later shot repeats. In [`films/fox.json`](films/fox.json),
shot 2 continues shot 1, shot 3 cuts to a raven, and shot 4 returns to the fox
from the last frame of shot 2.

## Checks

| Command | Checks |
|---------|--------|
| `./run.sh check_tp.py` | On CPU, that the transformer split over 2 ranks by [`parallel.py`](parallel.py) computes what it computes whole. |
| `./run.sh check_sp.py` | On CPU, that the transformer split over 2 × 2 ranks by [`parallel.py`](parallel.py), its weights by tensor parallelism and its video tokens by sequence parallelism, computes what it computes whole, with a timestep for the clip, with one per video token, and with two timesteps in place of one per token of two values (see [`timesteps.py`](timesteps.py)). |
| `./run.sh check_frame_split.py` | On CPU, that the video VAE split by frames over 4 ranks by [`parallel.py`](parallel.py) decodes what it decodes whole. |
| `./run.sh check_neuron.py` | On NeuronCores 0 and 1, that the transformer split over them computes what it computes on CPU, within the tolerance of the model support sweep. It takes the cores: run it when nothing else uses them. |
