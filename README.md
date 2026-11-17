
## Installation

Choose one installation route:

| Route | Use it when you want to… |
|---|---|
| [Docker](assets/warp_usage_docs/docker.md) | run a CUDA policy server, train, develop in a container, or deploy to an offline GPU host |
| [Native Python environment](#native-python-environment) | manage Python and CUDA dependencies directly on your machine |

Docker includes the project's Python environment. GPU hosts still need an
NVIDIA driver and NVIDIA Container Toolkit; see the guide's host requirements.

### Native Python environment

Create an environment with **conda**:

```bash
# Requires Python >= 3.10
conda create -n warp python=3.10
conda activate warp
```

or with **venv**:

```bash
# Requires Python >= 3.10 (check with `python3 --version`)
python3 -m venv .venv
source .venv/bin/activate
```

We recommend using PyTorch 2.7.1 with CUDA 12.8 (others may also work):

```bash
pip install torch==2.7.1 torchvision==0.22.1 torchaudio==2.7.1 --index-url https://download.pytorch.org/whl/cu128
```

Then install warp:

```bash
pip install -e .
```

> `deepspeed` ships as a source distribution and is compiled during `pip install`, so a C compiler (`gcc`) must be on the PATH.

<details>
<summary><b>Cosmos-Predict2.5 Extras (Optional)</b> — needed only for experiments with the <code>cosmos_predict25_2b</code> video backbone</summary>

With your environment activated:

```bash
git submodule update --init third_party/cosmos-predict2.5
bash scripts/install_cosmos_predict25.sh
```

The script installs the upstream cosmos packages into the active environment and compiles `transformer-engine` (CUDA toolkit with `nvcc` required), then automatically restores the package versions warp pins.

</details>

## Assets Preparation

The downloaders are interactive by default; every menu step also has a flag (`--name`, `--source`, `--root`, `--yes`, see `--help`) so they can run unattended, and the default storage location is resolved relative to the repository regardless of the working directory. Component downloaders store assets under
`assets/` and update the matching YAML path; the released-checkpoint downloader
keeps each checkpoint's self-contained config unchanged.

### 1. Video Backbone

~~~bash
python scripts/download_assets/download_video_backbone.py
~~~

**Supported video backbones**

<table>
<tr>
<td>Wan2.2-TI2V-5B ✅</td>
<td>Wan2.1-VACE-1.3B ✅</td>
<td>Wan2.1-I2V-14B-480P ✅</td>
</tr>
<tr>
<td>Cosmos-Predict2.5-2B ✅</td>
<td>Cosmos3-Edge ✅</td>
<td></td>
</tr>
</table>

Weights are saved under `assets/video_backbone_ckpt/` and the selected
`configs/model/video_backbone/` file is updated with the downloaded path.

### 2. Benchmark Data

~~~bash
python scripts/download_assets/download_benchmark_data.py
~~~

**Supported benchmarks**

<table>
<tr>
<td>RoboTwin2.0 ✅</td>
<td>RoboDojo ✅</td>
<td>RoboDojo-Real ✅</td>
</tr>
<tr>
<td>LIBERO ✅</td>
<td>VLABench ✅</td>
<td>EBench ✅</td>
</tr>
<tr>
<td>RoboCasa365 ✅</td>
<td>RoboCasa_GR1 ✅</td>
<td></td>
</tr>
</table>

Data is saved under `assets/benchmark_data/<benchmark>/`. Normalization
statistics are prepared when needed, and the selected dataloader configuration
is updated.

### 3. VLM Backbone (Optional)

Required only by tri_system:

~~~bash
python scripts/download_assets/download_vlm_backbone.py
~~~

**Supported VLM backbones**

<table>
<tr>
<td>Qwen3-VL-2B-Instruct ✅</td>
</tr>
</table>

Weights are saved under `assets/vlm_backbone_ckpt/`, and the selected configuration is updated.

### 4. Visual Encoders (Optional)

Required only for video backbones that use an external encoder:

~~~bash
python scripts/download_assets/download_visual_encoder.py
~~~

**Supported visual encoders**

<table>
<tr>
<td>DINOv3 ViT-B/16 ✅</td>
<td>V-JEPA 2.1 ViT-G/16 ✅</td>
</tr>
<tr>
<td>Wan2.2 VAE ✅</td>
<td>FLUX.2 VAE ✅</td>
</tr>
</table>

Weights are saved under `assets/visual_encoder_ckpt/`, and the selected encoder configuration is
updated.

### 5. Released warp Checkpoints

Use this downloader to obtain warp-Alpha releases or warp-Study
checkpoints from the warp collection:

~~~bash
python scripts/download_assets/download_warp_checkpoints.py
~~~

Checkpoints are saved under `assets/warp_ckpt/warp_alpha/` or
`assets/warp_ckpt/warp_study/<type>/`. Each checkpoint directory contains
its own config and can be deployed directly with:

~~~bash
bash scripts/deploy.sh <ckpt_dir_path>
~~~

For fine-tuning, set training.finetune_ckpt_path in `configs/train.yaml` to the
downloaded checkpoint directory. Benchmark data is still required.


## Quick Start

Quick Start provides a minimal end-to-end example: prepare the assets, train
or fine-tune a policy, deploy its checkpoint, and run a first inference check.

The example uses the DualSystem JointSelfAttention architecture, the
Wan2.2-TI2V-5B video backbone, and the Mutual attention mask.

| Component | Selection | Configuration |
|---|---|---|
| Architecture | dual_system / joint_self_attn | [`configs/model/dual_system.yaml`](configs/model/dual_system.yaml) |
| Video backbone | wan22_ti2v_5b | [`configs/model/video_backbone/wan22_ti2v_5b.yaml`](configs/model/video_backbone/wan22_ti2v_5b.yaml) |
| Attention mask | mutual | model.architecture.attention_mask_mode |
| Dataset | libero | [`configs/dataloader/libero.yaml`](configs/dataloader/libero.yaml) |

> **Resource recommendation:** We recommend 8 GPUs with 80 GB VRAM each for
> training. This configuration supports normal training for all architectures
> using Wan2.2-5B or smaller video backbones. More GPUs are better when
> available and can further improve training throughput.

### From Scratch Training

1. Download LIBERO and let the downloader update its dataloader configuration:

   ~~~bash
   python scripts/download_assets/download_benchmark_data.py
   ~~~

   Select LIBERO in the interactive menu.

2. Download Wan2.2-TI2V-5B:

   ~~~bash
   python scripts/download_assets/download_video_backbone.py
   ~~~

   Select Wan2.2-TI2V-5B and the desired model source.

3. Start a debug run with the complete model selection:

   ~~~bash
   bash scripts/train.sh \
     dataloader=libero \
     model=dual_system \
     model/video_backbone=wan22_ti2v_5b \
     model.architecture.variant=joint_self_attn \
     model.architecture.attention_mask_mode=mutual \
     training.debug=true
   ~~~

   training.debug=true runs 20 steps, saves at steps 10 and 20, and uses a
   constant learning rate. Check the run output, then set
   training.debug=false for normal training. Training defaults and CLI
   overrides are defined in [`configs/train.yaml`](configs/train.yaml).
   Debug outputs use training.output_path, whose default is
   `outputs/warp_checkpoints`.

4. Deploy the debug checkpoint and inspect one input-output cycle:

   ~~~bash
   bash scripts/deploy.sh <debug_ckpt_dir_path>
   ~~~

   Keep the server running, then open another terminal and run the two
   inference helpers. Deployment enables compile by default, so the first
   inference may take longer while compilation warms up; later requests are
   typically faster:

   ~~~bash
   python scripts/inference_test/inference_single_test.py \
     --server ws://127.0.0.1:8848 --test --state-dim 10

   python scripts/inference_test/inference_continuous_test.py \
     --server ws://127.0.0.1:8848 --test --state-dim 10
   ~~~

   The single-request helper checks ping, one prediction, and reset. The
   continuous helper sends repeated predictions over one connection and
   reports the returned action dimension and latency. Stop the deployment
   process after the checks.

### warp-α Fine-Tuning

1. Download LIBERO as shown above.

2. Download the warp-Alpha foundation checkpoint:

   ~~~bash
   python scripts/download_assets/download_warp_checkpoints.py
   ~~~

   Select warp_Alpha and warp-Alpha-Pretrain-Foundation-Model.
   Keep the resulting directory as <foundation_ckpt_dir_path>.

3. Start fine-tuning from that directory:

   ~~~bash
   bash scripts/train.sh \
     dataloader=libero \
     training.finetune_ckpt_path=<foundation_ckpt_dir_path>
   ~~~

   The default model configuration already matches the required setup above.
   You can set the same field in [`configs/train.yaml`](configs/train.yaml)
   instead of passing it on the command line. num_frames=33 and
   video_stride=4 in [`configs/dataloader/libero.yaml`](configs/dataloader/libero.yaml)
   produce the 32-step action horizon expected by the sampler; no separate
   action_chunk override is needed.

4. Deploy the resulting checkpoint directory:

   ~~~bash
   bash scripts/deploy.sh <ckpt_dir_path>
   ~~~

   Install and run the LIBERO client according to the
   [LIBERO evaluation guide](benchmarks/libero/README.md).


## warp Usage Guidance

warp is configured through composable Hydra YAML files. Select an architecture,
backbone, dataloader, and runtime behavior by changing configuration values or
overriding them on the command line. The task guides below are the maintained
entry points for using and extending the repository:

| Guide | Use it when you want to… |
|---|---|
| [Docker and offline deployment](assets/warp_usage_docs/docker.md) | build an image, run a first policy, train, develop with Compose, or deploy to an offline GPU host |
| [Training and deployment](assets/warp_usage_docs/train-and-deploy.md) | choose a model/dataloader, prepare assets, train, fine-tune, resume, or deploy a policy |
| [Architecture extension](assets/warp_usage_docs/architecture-extension.md) | extend a video, visual, VLM, action backbone, or WAM architecture |
| [Benchmark integration](assets/warp_usage_docs/benchmark-integration.md) | extend a dataloader and connect a benchmark client to the WebSocket protocol |
| [warp-α fine-tuning](assets/warp_usage_docs/warp-alpha-finetuning.md) | fine-tune the released foundation checkpoint to execute downstream task |

> 💡 **Agent tip:** Pick the guide that matches your task and feed it directly
> to your agent — less explaining, more building.

Installation and Assets Preparation above cover environment setup and model or
dataset downloads. Benchmark-specific environment and evaluation details remain
in the benchmarks directory.

## Projects Based on warp

- **[LD4WAM](https://github.com/stubborn111/LD4WAM)** — *LD4WAM: Learning Latent Dynamics from Human Videos for World Action Models* ([arXiv:2608.22403](https://arxiv.org/abs/2608.22403)). Learns a Latent Dynamics Model (LDM) from human videos and uses it to build a World Dynamics Action Model (WDAM) — a Wan2.2 video DiT coupled with a latent-dynamics expert and an action expert — on top of the warp framework, evaluated on RoboTwin 2.0.

If you build on warp, feel free to open a PR to add your project here.

## Star History

<p align="center">
  <a href="https://star-history.com/#warp-Official/warp&Date">
    <picture>
      <source media="(prefers-color-scheme: dark)" srcset="https://api.star-history.com/svg?repos=warp-Official/warp&amp;type=Date&amp;theme=dark">
      <source media="(prefers-color-scheme: light)" srcset="https://api.star-history.com/svg?repos=warp-Official/warp&amp;type=Date">
      <img src="https://api.star-history.com/svg?repos=warp-Official/warp&amp;type=Date" alt="Star history of warp-Official/warp" width="70%">
    </picture>
  </a>
</p>

## License

warp is released under the [Apache License 2.0](LICENSE).


