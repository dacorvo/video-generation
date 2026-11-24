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
./run.sh generate.py --two-stage  # 10 s at 2048x1024, transformer on NeuronCores 0 to 31, video decoder on 16 to 31
./run.sh generate.py --cpu        # all on CPU, to compare
```

With `--two-stage`, [`generate.py`](generate.py) follows the two-stage recipe of
the model card: 8 steps at 1024x512, the x2 latent upsampler on CPU, then 3
steps at 2048x1024.

[`generate.py`](generate.py) makes the shots of a JSON file (see
[Shots](#shots)) one after the other, without loading the models again. Without
a file, it makes one shot of a red fox. `--seed` sets the seed of the shots that
have none.

It writes each clip to `outputs/`, named after the file and the size of the
clip, or `video-gen-demo-WIDTHxHEIGHT` without a file: an mp4 with its audio, 4
of its frames as a png and, with `--npz`, its frames and audio as an npz. With
several shots, the name of each clip ends with the name of its shot, and the
clip that ends with `-full` is the shots one after the other:

```bash
./run.sh generate.py --two-stage FILE.json  # outputs/FILE-2048x1024-SHOT.mp4 for each shot, and outputs/FILE-2048x1024-full.mp4
```

The examples, with their JSON files, are in the
[dacorvo/video-generation](https://huggingface.co/buckets/dacorvo/video-generation)
bucket.

## Shots

A JSON file is a list of shots, each with a name and a prompt. Shots refer to
each other by name, so that shots can be moved, added or removed by hand
without the others pointing at the wrong shot:

```json
[
  {"name": "trot", "prompt": "..."},
  {"name": "edge", "prompt": "...", "from": {"shot": "trot", "frames": 25}}
]
```

A shot can start from frames, which it keeps as its first frames:

- From the last frames of an earlier shot, `"from": {"shot": "trot", "frames": 25}`.
  From its last frame, a shot returns to its subject after a cut, with the same
  look, as prompts alone do not keep it. From its last 25 frames (1 s), it
  continues its motion.
- From an image, `"from": {"image": "frame.png"}`, such as a frame of a camera:
  the prompt says what happens next.
- From the last frames of a video at 24 frames per second,
  `"from": {"video": "start.mp4", "frames": 25}`, such as the last second a
  camera filmed: the shot also takes the speed of what moves in it.

The paths of images and videos are relative to the JSON file. The pipeline
scales them to cover the clip and keeps their middle.

A shot can end on the first frames of a later shot,
`"to": {"shot": "scream", "frames": 9}`, which the pipeline takes as keyframes,
so that the shot leads into it. The process then makes the later shot first.

A shot can have its own seed, `"seed": 3`, in place of `--seed`, to keep the
take of the shot picked among several. It can have a caption, drawn on its
frames from `start` to `end`, in seconds into the shot,
`"caption": {"text": "GOOOAAAATTT", "start": 4.4, "end": 6.4}`, and be trimmed
to the time range from `start` to `end`, `"trim": {"start": 4.5, "end": 8.0}`,
as all shots come out the same length.

The numbers of frames are one more than a multiple of 8, as the VAE encodes the
frames after the first by 8. The film leaves out the frames a shot starts from
when they repeat an earlier shot, and the frames of a shot from its keyframes
on, which the later shot repeats. As the model generates the sound of each shot
on its own, the background of one shot can be faint and the next one loud: the
film raises its quiet parts toward a common level, and leaves its loud sounds
as they are.

## Checks

| Command | Checks |
|---------|--------|
| `./run.sh check_tp.py` | On CPU, that the transformer split over 2 ranks by [`parallel.py`](parallel.py) computes what it computes whole. |
| `./run.sh check_sp.py` | On CPU, that the transformer split over 2 × 2 ranks by [`parallel.py`](parallel.py), its weights by tensor parallelism and its video tokens by sequence parallelism, computes what it computes whole, with a timestep for the clip, with one per video token, and with two timesteps in place of one per token of two values (see [`timesteps.py`](timesteps.py)). |
| `./run.sh check_frame_split.py` | On CPU, that the video VAE split by frames over 4 ranks by [`parallel.py`](parallel.py) decodes what it decodes whole. |
| `./run.sh check_neuron.py` | On NeuronCores 0 and 1, that the transformer split over them computes what it computes on CPU, within the tolerance of the model support sweep. It takes the cores: run it when nothing else uses them. |
