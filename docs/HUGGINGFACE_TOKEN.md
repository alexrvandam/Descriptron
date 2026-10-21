# Hugging Face access token (DINOv3 and SAM 3 weights)

Several Descriptron tools use model weights that Meta publishes on Hugging Face behind a licence
("gated" models). They download automatically the first time a tool needs them, but only if Hugging Face
knows who you are and that you accepted the licence. That takes three things, each done **once**:
a free account, access to each model, and a token on your computer.

> **The ViT-L model to use for DINOMask** (and BioRAG Annotate / BioRAG-Probe):
> **[facebook/dinov3-vitl16-pretrain-lvd1689m](https://huggingface.co/facebook/dinov3-vitl16-pretrain-lvd1689m)**
> - DINOv3 ViT-L/16, distilled, trained on LVD-1689M (web images). Not the `sat493m` (satellite) ViT-L.

## Which tools need which model

| tool | model (Hugging Face id) | size | page to request access |
|---|---|---|---|
| **DINOMask** (`descriptron-dinomask`), BioRAG Annotate (GUI tab 6), BioRAG-Probe | `facebook/dinov3-vitl16-pretrain-lvd1689m` (ViT-L/16) | 1.2 GB | https://huggingface.co/facebook/dinov3-vitl16-pretrain-lvd1689m |
| **DINOLand** (`descriptron-dinoland`), default model | `facebook/dinov3-vitb16-pretrain-lvd1689m` (ViT-B/16) | 0.3 GB | https://huggingface.co/facebook/dinov3-vitb16-pretrain-lvd1689m |
| DINOLand with `--model ...vitl16...` | the ViT-L/16 above | | |
| **SAM 3** (*Find all instances (SAM 3)* button, `descriptron-sam3`) | `facebook/sam3` | 3.3 GB | https://huggingface.co/facebook/sam3 |

SAM2, SAM2-PAL and the Florence-2 scale-bar reader are **not** gated; they need no token.

Note for DINOMask: pass `--model facebook/dinov3-vitl16-pretrain-lvd1689m` as in the
[DINOMask recipe](../descriptron/dinomask/README.md). Without `--model` the script falls back to the much smaller
ViT-S/16; all our reported DINOMask results used the ViT-L.

## Step 1 - make a Hugging Face account

Sign up at https://huggingface.co/join (free) and confirm your e-mail address.

## Step 2 - request access to each model you need

Open the model page from the table above while logged in, read the licence, fill in the short form and submit.
Do this for every model you will use (for DINOMask and DINOLand together: both DINOv3 pages).

Access is granted by Meta, usually within minutes to hours. You can see the status of every request at
https://huggingface.co/settings/gated-repos - wait until it says *accepted* before going on to step 4; a token
cannot download a model you have not been given access to.

Use the `lvd1689m` models listed above (trained on 1.7 billion web images). The `sat493m` DINOv3 models are trained
on satellite images and are not what Descriptron was tested with.

## Step 3 - create an access token

1. Go to https://huggingface.co/settings/tokens and click **Create new token**.
2. Choose the token type **Read**. (A *fine-grained* token also works if you tick
   *Read access to contents of all public gated repos you can access*.)
3. Give it a name (e.g. `descriptron-laptop`) and create it.
4. Copy the token - it starts with `hf_`. Treat it like a password: do not paste it into e-mails, code, notebooks or
   GitHub. If it leaks, delete it on the same page and make a new one.

## Step 4 - put the token on your computer (choose ONE way)

### A. Log in once with the `hf` command (recommended - every tool finds it)

In the terminal where you use Descriptron (activate its environment first):

    hf auth login

Paste the token when asked (nothing is shown while you paste - that is normal) and answer `n` to the question about
git credentials. The token is saved in your user folder (`~/.cache/huggingface/token`) and is used by every
Python environment of your user account, including the separate SAM 3 environment.

- If `hf` is not found, use the older name: `huggingface-cli login`.
- SAM 3 installed with pipx: `pipx run --spec huggingface_hub hf auth login` does the same.
- Check it worked: `hf auth whoami` prints your user name.

### B. An environment variable

Linux / macOS - for the current terminal:

    export HF_TOKEN=hf_xxxxxxxxxxxxxxxxx

To keep it, add that line to `~/.bashrc` (Linux) or `~/.zshrc` (macOS) and open a new terminal.

Windows - Command Prompt, saved for all new terminals (open a new one afterwards):

    setx HF_TOKEN hf_xxxxxxxxxxxxxxxxx

Windows - PowerShell, current window only: `$env:HF_TOKEN = "hf_xxxxxxxxxxxxxxxxx"`

### C. Descriptron's key store

    descriptron-keys --set HF_TOKEN

saves it in `~/.config/descriptron/credentials` (private to your user). DINOLand also asks for the token by itself the
first time it needs one and saves it there. **In 2.8.0 only DINOLand and the GUI read this file; DINOMask, BioRAG
Annotate and BioRAG-Probe do not** - for those use A or B.

### Docker

Pass the token into the container with `-e HF_TOKEN` (after `export HF_TOKEN=...` as in B), and keep the
downloaded weights in the `descriptron-weights` volume so they download only once:

    export HF_TOKEN=hf_xxxxxxxxxxxxxxxxx
    docker run --rm --gpus all -e HF_TOKEN -v "$PWD:/data" -v descriptron-weights:/weights \
      ghcr.io/alexrvandam/descriptron:latest dinomask --model facebook/dinov3-vitl16-pretrain-lvd1689m ...

The saved-key route (`-v "$HOME/.config/descriptron:/config"` and `keys --set HF_TOKEN`, see
[DOCKER_RECIPE.md](DOCKER_RECIPE.md)) works for DINOLand; use `-e HF_TOKEN` for DINOMask.

## Step 5 - test the download (optional, about 1 minute)

With the Descriptron environment active:

    python -c "from transformers import AutoModel; AutoModel.from_pretrained('facebook/dinov3-vitl16-pretrain-lvd1689m'); print('DINOv3 ViT-L OK')"

After the first successful download the weights are cached (`~/.cache/huggingface/hub`, or the Docker volume) and
no token is needed again for that model - also not offline (`export HF_HUB_OFFLINE=1`).

## If something goes wrong

| message | cause and fix |
|---|---|
| `401 Client Error` / `Repository not found` / `Invalid user token` | no token, or a mistyped one: repeat step 4 (`hf auth whoami` should show your name) |
| `403` / `Access to model ... is restricted` / `gated repo` | the token is fine but access is not granted (yet) for **this** model: check https://huggingface.co/settings/gated-repos, request access on the model page (step 2) |
| a fine-grained token gets `403` although access was accepted | edit the token and tick *Read access to contents of all public gated repos you can access*, or make a *Read* token instead |
| `model type 'dinov3_vit' ... Transformers does not recognize this architecture` | `transformers` is too old for DINOv3 (needs 4.56 or newer): `pip install -U "transformers>=4.56"`, or `pip install --upgrade descriptron` |
| DINOLand works but DINOMask is refused | the token is only in Descriptron's key store (C); use A or B |
| a shared lab machine or cluster without internet on the compute nodes | download once on a node with internet (step 5), point everyone to the same cache with `HF_HOME=/shared/path`, and run with `HF_HUB_OFFLINE=1`; see [SHARED_VM_RECIPE.md](SHARED_VM_RECIPE.md) |
