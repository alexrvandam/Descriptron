# Hosting Descriptron on a shared virtual machine

For IT and research-computing staff who want to run Descriptron for a group of
taxonomists on one institutional VM (shared memory, shared GPU), using the
Docker image. For a single user on their own computer, see
[DOCKER_RECIPE.md](DOCKER_RECIPE.md) instead.

**In short:** one Linux VM with Docker and the NVIDIA container toolkit, one
Descriptron image, one shared read-only folder of model weights, one project
folder per research group, and every user running containers under their own
Unix account. Nothing needs to run as a permanent service.

---

## 1. What runs where

Descriptron is a set of command-line programs plus a desktop annotation GUI. The
image (`ghcr.io/alexrvandam/descriptron`) contains all of them in four conda
environments, including SAM2 and Detectron2, which cannot be installed from PyPI.

| task | needs a GPU? | notes |
|---|---|---|
| annotation GUI (drawing masks and landmarks, SAM2-assisted) | recommended | Tkinter desktop app; needs a remote desktop (section 6) |
| SAM2-PAL mask propagation, fine-tuning | **yes** | the heaviest job; validated on a 12 GB GPU |
| detector training (Mask R-CNN / torchvision, Detectron2) | **yes** | long runs (hours) |
| DINOLand landmark transfer | no | runs on CPU: 80 wings in about 4 minutes |
| measurements, matrix, key, delimitation, audits, exports | no | CPU only |
| species descriptions (treatments) | no GPU, but a **language model** | section 7 |

---

## 2. Hardware

What the published analyses were run on (one workstation): 32 CPU threads, 32 GB
RAM, one NVIDIA RTX 3500 Ada with 12 GB of GPU memory.

Suggested for a VM shared by a group:

| resource | minimum | comfortable for 5–10 users |
|---|---|---|
| CPU | 16 cores | 32+ cores |
| RAM | 32 GB | 64–128 GB |
| GPU | 1 × NVIDIA, 12 GB | 1–2 × NVIDIA, 24–48 GB (e.g. A40, L40S, A6000), or an A100/H100 split with MIG |
| system disk | 100 GB (image 33 GB + weights about 5 GB + scratch) | 200 GB |
| project storage | size of the image collections | plan in TB for micro-CT; backed up |

NVIDIA architectures confirmed to work: Ampere, Ada, Hopper. Blackwell (RTX 50xx,
B100/B200) is **not** supported by the PyTorch builds in the image.

---

## 3. Software on the VM

- Any recent Linux (the image is based on Ubuntu 22.04; developed on Ubuntu 24.04)
- NVIDIA driver R550 or newer
- Docker Engine (or Podman) and the **NVIDIA Container Toolkit**

Check that containers see the GPU:

```bash
docker run --rm --gpus all ubuntu nvidia-smi
```

---

## 4. Install (once)

```bash
# the image (about 33 GB on disk); pin a version for reproducible projects
docker pull ghcr.io/alexrvandam/descriptron:2.0.3

# shared folders
sudo mkdir -p /srv/descriptron/{weights,checkpoints,projects}
```

**Model weights.** They are not in the image.

- **SAM2 checkpoints** (about 1.4 GB for all sizes): download Meta's official files
  into `/srv/descriptron/checkpoints` (e.g. `sam2_hiera_large.pt`) with
  `download_ckpts.sh` from the SAM2 repository.
- **DINOv3** (DINOLand): gated on Hugging Face. An administrator accepts Meta's
  licence once, downloads `facebook/dinov3-vitb16-pretrain-lvd1689m` (0.3 GB) and
  optionally `facebook/dinov3-vitl16-pretrain-lvd1689m` (1.2 GB) into a shared
  Hugging Face cache, `/srv/descriptron/weights/huggingface`.
- Everything else (EasyOCR, Florence-2, torchvision backbones) downloads on first
  use into the weights folder, so make it writable by the Descriptron group the
  first time, then read-only.

**Smoke test:**

```bash
docker run --rm ghcr.io/alexrvandam/descriptron:2.0.3 help
docker run --rm ghcr.io/alexrvandam/descriptron:2.0.3 mcp --check     # lists 62 + 11 programs
```

---

## 5. Users, projects and permissions

- One Unix group per research group; one project folder each:
  `/srv/descriptron/projects/<group>/`, mode `2770`, owned by that group.
  **Some collections are confidential** (unpublished species, locality data under
  agreements); group permissions keep them apart.
- Users run containers **as themselves**, so output files belong to them:

```bash
docker run --rm --gpus all --user "$(id -u):$(id -g)" \
  -v /srv/descriptron/projects/mygroup:/data \
  -v /srv/descriptron/checkpoints:/ckpt:ro \
  -v /srv/descriptron/weights:/weights \
  ghcr.io/alexrvandam/descriptron:2.0.3 sam2-pal \
    --template_image /data/refs/template.png --template_json /data/refs/annotations.json \
    --image_dir /data/targets --output_dir /data/out_sam2pal \
    --sam2_checkpoint /ckpt/sam2_hiera_large.pt --sam2_config sam2_hiera_l.yaml
```

- To let users run containers without root, add them to the `docker` group (this
  is root-equivalent on the host) or use **rootless Docker / Podman**, which is the
  safer choice on a shared machine.
- A wrapper script in `/usr/local/bin/descriptron` that adds the `--user`, the
  mounts and the image tag saves users from typing them.

---

## 6. The annotation GUI on a VM

The GUI is a desktop (Tkinter/X11) program. On a VM users reach it through a
remote desktop — X2Go, xrdp, or a browser-based noVNC/Xpra session — and start it
from that desktop:

```bash
docker run --rm --gpus all --user "$(id -u):$(id -g)" \
  -e DISPLAY=$DISPLAY -v /tmp/.X11-unix:/tmp/.X11-unix \
  -v /srv/descriptron/projects/mygroup:/data -v /srv/descriptron/weights:/weights \
  ghcr.io/alexrvandam/descriptron:2.0.3 gui
```

We run the GUI on local Linux desktops; a remote-desktop setup is standard
X11 forwarding but has not been tested by us on a VM. Plain `ssh -X` works but is
slow for large images.

---

## 7. The language model for species descriptions

Everything up to the matrix, the key and the evidence sheets needs **no** model.
Writing the prose of the species descriptions needs a vision-language model; the
pipeline flag is `--llm-backend`:

| backend | inside the image | what the host provides |
|---|---|---|
| `api` | **yes** (Anthropic SDK installed) | an API key in an environment variable (`--api-key-env NAME`); `--api-base-url` for any Anthropic-compatible endpoint |
| `claude-code` | no (the Claude Code CLI is not in the image) | run the pipeline outside the container on a machine with Claude Code installed and logged in |
| `none` | yes | stops cleanly after the matrix, key and evidence sheets |

The published descriptions were generated with Claude (Sonnet 5). Other models can
be connected through `--api-base-url` but have not been validated.

Keys belong to users or projects, never in the image or in shared scripts: pass
them per run (`-e ANTHROPIC_API_KEY`) from the user's own environment.

---

## 8. Sharing the GPU

- SAM2-PAL fine-tuning and detector training each want a GPU to themselves;
  prediction runs are shorter. Expect two such jobs on one 12 GB GPU to run out of memory.
- Simple: one GPU job at a time, agreed in the group, or `CUDA_VISIBLE_DEVICES`
  per user when the VM has several GPUs.
- Better for larger groups: a small queue (Slurm, or even a lock file in the
  wrapper script), or MIG slices on an A100/H100.
- DINOLand and the whole analysis pipeline run on CPU and can run alongside GPU jobs.

---

## 9. Updates and reproducibility

- New versions are published as new image tags (`2.0.3`, …) and `latest`.
  `docker pull` fetches only the changed layers.
- Pin the tag in each project's scripts and note it with the results, so an
  analysis can be rerun with the exact image it used.
- Annotation guidance for SAM2-PAL and DINOLand:
  [SAM2-PAL & DINOLand annotation SOP](SAM2PAL_DINOLand_Annotation_SOP.md).

---

## 10. Optional: use from Claude (MCP)

Users with Claude Code can drive the programs on the VM through the MCP server in
the image (`descriptron mcp`); see the main README, section *Use Descriptron from
Claude*.

---

Questions: Alex Van Dam (Museum für Naturkunde Berlin), via the GitHub issues of
this repository. Please cite Descriptron as described in the main README.
