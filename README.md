# LTX-2.5 on Trainium 3

Generates video with audio with [LTX-2.5](https://huggingface.co/Lightricks/LTX-2.5-Diffusers)
on a trn3.48xlarge, its transformer split over the NeuronCores.

It makes two kinds of clips: [films](#films), whose shots follow one another,
and [continuations](#continuations), clips that go on from a frame of a real
camera, as training videos for robots, cars and drones.

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
./run.sh generate.py --two-stage  # 10 s at 2048x1024, transformer on NeuronCores 0 to 31, video decoder on 16 to 31
./run.sh generate.py --cpu        # all on CPU, to compare
```

With `--two-stage`, [`generate.py`](generate.py) follows the two-stage recipe of
the model card: 8 steps at 1024x512, the x2 latent upsampler on CPU, then 3
steps at 2048x1024.

What [`generate.py`](generate.py) generates is a JSON file, a list of shots,
each with a name and a prompt: a [film](#films), or
[continuations](#continuations) of a frame. Without a file, it makes one shot
of a red fox. It makes the shots one after the other, without loading the
models again. `--seed` sets the seed of the shots that have none.

It writes each clip to `outputs/`, named after the file and the size of the
clip, such as `goat-2048x1024`, or `video-gen-demo-2048x1024` without a file:
an mp4 with its audio, 4 of its frames as a png and, with `--npz`, its frames
and audio as an npz. With several shots, the name of each clip ends with the
name of its shot, such as `goat-2048x1024-climb`, and the clip that ends with
`-full` is the film of the shots, one after the other.

Each shot refers to the others by their names, so that shots can be moved,
added or removed by hand without the others pointing at the wrong shot.

## Films

A film is a file of shots that follow one another. The films made with
[`generate.py`](generate.py), their JSON files and their mp4s, are in the
[dacorvo/video-generation](https://huggingface.co/buckets/dacorvo/video-generation)
bucket. For instance, for the goat film:

```bash
curl -LO https://huggingface.co/buckets/dacorvo/video-generation/resolve/goat.json
./run.sh generate.py --two-stage goat.json  # outputs/goat-2048x1024-full.mp4, and its shots goat-2048x1024-valley.mp4...
```

A shot can start from the last frames of an earlier shot, which it keeps as its
first frames:

```json
{"name": "lake", "prompt": "...", "from": {"shot": "edge", "frames": 1}}
```

- From the last frame of a shot, a shot returns to its subject after a cut,
  with the same look, as prompts alone do not keep it.
- From the last 25 frames (1 s), a shot continues the motion of the shot.

A shot can also end on the first frames of a later shot, which the pipeline
takes as keyframes, so that the shot leads into it. The process then makes the
later shot first:

```json
{"name": "climb", "prompt": "...", "to": {"shot": "scream", "frames": 9}}
```

A shot can have its own seed, in place of `--seed`, to keep the take of the shot
picked among several:

```json
{"name": "valley", "prompt": "...", "seed": 3}
```

A shot can have a caption, drawn on its frames from `start` to `end`, in seconds
into the shot:

```json
{"name": "scream", "prompt": "...", "caption": {"text": "GOOOAAAATTT", "start": 4.4, "end": 6.4}}
```

A shot can be trimmed to the time range from `start` to `end`, in seconds into
the shot, as all shots come out the same length:

```json
{"name": "eye", "prompt": "...", "trim": {"start": 4.5, "end": 8.0}}
```

The number of frames is one more than a multiple of 8, as the VAE encodes the
frames after the first by 8. The film leaves out the frames a shot starts
from, which repeat the earlier shot, and the frames of a shot from its
keyframes on, which the later shot repeats. As the model generates the sound of
each shot on its own, the background of one shot can be faint and the next one
loud: the film raises its quiet parts toward a common level, and leaves its
loud sounds as they are.

In the bucket, in `fox.json`, `edge` continues `trot`, `raven` cuts to a
raven, and `lake` returns to the fox from the last frame of `edge`. In
`goat.json`, `climb` ends on the first frames of `scream`, `eye` is an insert
of 3.5 s trimmed from its shot, and `scream` has a caption.

## Continuations

Robots, cars and drones learn from video of what their cameras see. A world
model makes more of it: from a frame of a camera, it generates what the camera
may see next, including the rare cases that are costly or dangerous to film,
such as a deer jumping onto a snowy road. LTX calls LTX-2.5 a world model.

A shot can start from an image, such as a frame of a camera, which it keeps as
its first frame, and its prompt says what happens next. The path of the image
is relative to the JSON file. Several shots can start from the same frame, each
with its own future, such as what happened and a rare case:

```json
[
  {"name": "happened", "prompt": "...", "from": {"image": "boreas-snow.png"}},
  {"name": "deer", "prompt": "...", "from": {"image": "boreas-snow.png"}}
]
```

```bash
./run.sh generate.py --two-stage boreas-snow.json  # outputs/boreas-snow-2048x1024-happened.mp4 and -deer.mp4, and -full.mp4, both one after the other
```

The pipeline scales the image to cover the clip and keeps its middle: a frame
of another shape loses its sides, or its top and bottom.

Here the prompt drives the clip, not the commands of a robot or a car, which a
world model for training follows. To build one, a team fine-tunes the
[pre-trained checkpoint of LTX-2.5](https://huggingface.co/Lightricks/LTX-2.5-Pre-Trained)
on the frames and actions of its own robots: LTX calls that checkpoint "a
foundation to build from, not an out-of-the-box robotics or autonomous-driving
system" ([LTX](https://ltx.io/model/capabilities/world-model)).

## Checks

| Command | Checks |
|---------|--------|
| `./run.sh check_tp.py` | On CPU, that the transformer split over 2 ranks by [`parallel.py`](parallel.py) computes what it computes whole. |
| `./run.sh check_sp.py` | On CPU, that the transformer split over 2 × 2 ranks by [`parallel.py`](parallel.py), its weights by tensor parallelism and its video tokens by sequence parallelism, computes what it computes whole, with a timestep for the clip, with one per video token, and with two timesteps in place of one per token of two values (see [`timesteps.py`](timesteps.py)). |
| `./run.sh check_frame_split.py` | On CPU, that the video VAE split by frames over 4 ranks by [`parallel.py`](parallel.py) decodes what it decodes whole. |
| `./run.sh check_neuron.py` | On NeuronCores 0 and 1, that the transformer split over them computes what it computes on CPU, within the tolerance of the model support sweep. It takes the cores: run it when nothing else uses them. |
