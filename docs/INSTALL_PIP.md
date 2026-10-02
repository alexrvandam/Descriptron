# Installing Descriptron with pip (Windows, macOS, Linux)

Step by step, for a computer that has nothing installed yet. It takes about 15 minutes, most of it downloads.

You will install four things, all into **one Python environment** (a folder that keeps Descriptron's packages
apart from anything else on the computer):

| what | where it comes from | where it goes |
|---|---|---|
| PyTorch | PyTorch's website (pip) | the Descriptron environment |
| **SAM2 library** (the segmentation code) | Meta's GitHub (pip) | the Descriptron environment |
| Descriptron | PyPI (pip) | the Descriptron environment |
| **SAM2 model file** `sam2_hiera_large.pt` (898 MB) | Meta, downloaded by the GUI the first time it starts | a cache folder in your home folder (step 6) |

You never copy the SAM2 library anywhere by hand: `pip` puts it in the environment. The model file is downloaded
for you; you only place it yourself if you already have it (step 6).

---

## 1. Install Python 3.12 and Git (once)

**Windows**
- Python: https://www.python.org/downloads/ - download Python **3.12**, run the installer and **tick "Add python.exe
  to PATH"** on the first screen.
- Git (optional - step 4 has a no-Git option): https://git-scm.com/downloads/win - keep the default PATH option
  "Git from the command line and also from 3rd-party software", and open a NEW Command Prompt afterwards.
  Details and fixes: [INSTALL_GIT.md](INSTALL_GIT.md).

**macOS**
- Python: https://www.python.org/downloads/ - the **3.12** macOS installer.
- Git: open Terminal and type `git --version`; if it is missing, macOS offers to install it.

**Linux (Debian/Ubuntu)**

    sudo apt install python3.12 python3.12-venv python3-tk git

## 2. Make the Descriptron environment (once)

Open a terminal: **Windows** - Start menu, type `cmd`, open "Command Prompt". **macOS** - Terminal.

**Windows**

    py -3.12 -m venv %USERPROFILE%\descriptron-env
    %USERPROFILE%\descriptron-env\Scripts\activate

**macOS / Linux**

    python3.12 -m venv ~/descriptron-env
    source ~/descriptron-env/bin/activate

The prompt now starts with `(descriptron-env)`. **Every command below must be typed in a terminal where the
environment is active** - that is what puts everything in the same place.

## 3. Install PyTorch

**Windows or Linux with an NVIDIA graphics card** (much faster):

    pip install torch==2.10.0 torchvision==0.25.0 --index-url https://download.pytorch.org/whl/cu128

**macOS, or no NVIDIA card:**

    pip install torch==2.10.0 torchvision==0.25.0

(On Windows, plain `pip install torch` gives a CPU-only PyTorch, so use the first command if there is an NVIDIA
card.)

## 4. Install the SAM2 library (from Meta's GitHub)

SAM2 is not on PyPI, so it is installed from Meta's GitHub. `SAM2_BUILD_CUDA=0` skips an optional compiled part
that the GUI does not use, so no C++ compiler is needed.

**Windows (Command Prompt)**

    set SAM2_BUILD_CUDA=0
    pip install git+https://github.com/facebookresearch/sam2.git

(In PowerShell the first line is `$env:SAM2_BUILD_CUDA="0"`.)

**macOS / Linux**

    SAM2_BUILD_CUDA=0 pip install git+https://github.com/facebookresearch/sam2.git

**No Git (or "git is not recognized")?** Install the same thing from GitHub's zip download instead - replace the
`pip install git+...` line with:

    pip install https://github.com/facebookresearch/sam2/archive/refs/heads/main.zip

(keep the `SAM2_BUILD_CUDA=0` line before it). To put Git on the PATH instead, see [INSTALL_GIT.md](INSTALL_GIT.md).

## 5. Install Descriptron

    pip install descriptron

## 6. Start the GUI - and the SAM2 model file

    descriptron-gui

The first time, it asks to download the SAM2 model file (`sam2_hiera_large.pt`, 898 MB) from Meta. Click **Yes**;
the terminal shows the progress, and the GUI opens when it is done. The file is saved here and found
automatically from then on:

| | the SAM2 model file is saved as |
|---|---|
| Windows | `C:\Users\<your name>\.cache\descriptron\sam2\sam2_hiera_large.pt` |
| macOS | `/Users/<your name>/.cache/descriptron/sam2/sam2_hiera_large.pt` |
| Linux | `/home/<your name>/.cache/descriptron/sam2/sam2_hiera_large.pt` |

**Already have `sam2_hiera_large.pt`** (or no internet on that computer)? Either copy it into the folder above
(make the folders if they do not exist), or tell Descriptron where it is:

    setx DESCRIPTRON_SAM2_CHECKPOINT "D:\models\sam2_hiera_large.pt"              (Windows; open a new terminal after)
    export DESCRIPTRON_SAM2_CHECKPOINT=/path/to/sam2_hiera_large.pt               (macOS / Linux; add to ~/.zshrc or ~/.bashrc to keep it)

Direct download link, if you want to fetch it yourself:
https://dl.fbaipublicfiles.com/segment_anything_2/072824/sam2_hiera_large.pt

## Every time after that

Open a terminal, activate the environment, start the GUI:

    %USERPROFILE%\descriptron-env\Scripts\activate          (Windows)
    source ~/descriptron-env/bin/activate                   (macOS / Linux)
    descriptron-gui

You can drag an image or a folder of images onto the window, or start with one: `descriptron-gui D:\photos\wings`.

## Optional: SAM 3 (find every seta at once)

SAM 3 lives in its own environment. With the Descriptron environment **not** active:

    pip install pipx
    pipx ensurepath                         (then open a new terminal)
    pipx install --python python3.12 descriptron-sam3

Request access at https://huggingface.co/facebook/sam3, create a read token in your Hugging Face settings, and log
in once: `pipx run --spec huggingface_hub hf auth login`. The GUI's dark green **Find all instances (SAM 3)**
button then finds it.

## If something goes wrong

| message | fix |
|---|---|
| `SAM2 is not installed` / `No module named 'sam2'` | step 4 was not run in the active environment: activate it (step 2) and repeat step 4 |
| `git` is not recognised / `Cannot find command 'git'` | use the zip line in step 4 (no Git needed), or put Git on the PATH: [INSTALL_GIT.md](INSTALL_GIT.md) |
| `py` / `python3.12` not found | Python 3.12 is not installed, or (Windows) "Add python.exe to PATH" was not ticked: reinstall Python |
| `tkinter is missing` (Linux) | `sudo apt install python3-tk` |
| `descriptron-gui` not found | the environment is not active (step 2) |
| the GUI is slow | no NVIDIA card or the CPU PyTorch was installed: repeat step 3 with the NVIDIA command |
