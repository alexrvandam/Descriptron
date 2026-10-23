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

**Which Python, and where to get it.** Use **Python 3.12**. Newer versions (3.13, 3.14, 3.15) are not yet covered by
the PyTorch wheels Descriptron is tested with, so `pip install descriptron` may fail on them. Having another Python
installed as well does no harm: you choose 3.12 when you make the environment (step 2).

Two things that confuse almost everyone on python.org:
- **3.12.10 is the last 3.12 with ready-made installers** for Windows and macOS. Later 3.12 releases (3.12.11,
  3.12.12 ... 3.12.15) are security fixes published **only as source code**: a `Python-3.12.x.tgz` / `.tar.xz`
  file. There is **no .exe or .pkg inside it** - it has to be compiled, which is only practical on Linux. On Windows
  and macOS use 3.12.10; the differences are security patches that do not affect Descriptron.
- The front page of python.org and the **Python Install Manager** offer the **newest** Python by default, not 3.12.

**Windows**
- Python, either way:
  - **Installer:** https://www.python.org/ftp/python/3.12.10/python-3.12.10-amd64.exe
    (release page: https://www.python.org/downloads/release/python-31210/ , *Files*, "Windows installer (64-bit)").
    Run it and **tick "Add python.exe to PATH"** at the bottom of the first screen, then *Install Now*.
  - **Python Install Manager** (the newer `py` tool from python.org or the Microsoft Store): it installs the newest
    Python unless you ask for 3.12. In a Command Prompt:

        py install 3.12
        py list

    `py list` shows every installed version; 3.12 must be among them.
- Check, in a NEW Command Prompt: `py -3.12 --version` should print `Python 3.12.x`.
- Git (optional - step 4 has a no-Git option): https://git-scm.com/downloads/win - keep the default PATH option
  "Git from the command line and also from 3rd-party software", and open a NEW Command Prompt afterwards.
  Details and fixes: [INSTALL_GIT.md](INSTALL_GIT.md).

**macOS**
- Python: https://www.python.org/downloads/release/python-31210/ - *Files*, "macOS 64-bit universal2 installer"
  (`python-3.12.10-macos11.pkg`). After installing, open *Applications > Python 3.12* and double-click
  **Install Certificates.command** - without it, downloads of model weights fail with SSL certificate errors.
- Check in Terminal: `python3.12 --version`.
- Git: open Terminal and type `git --version`; if it is missing, macOS offers to install it.

**Linux (Debian/Ubuntu)**

    sudo apt install python3.12 python3.12-venv python3-tk git

If your distribution has no `python3.12` package, the simplest route is Miniconda
(https://docs.conda.io/en/latest/miniconda.html): `conda create -n descriptron python=3.12`, then
`conda activate descriptron` and continue at step 3 (it replaces step 2 and includes tkinter).
Compiling from the python.org source `.tgz` also works, but install the build libraries **first** - without `tk-dev`
the new Python has no tkinter and the GUI cannot start:

    sudo apt install build-essential zlib1g-dev libncurses5-dev libgdbm-dev libnss3-dev libssl-dev \
      libreadline-dev libffi-dev libsqlite3-dev libbz2-dev liblzma-dev tk-dev uuid-dev
    tar -xzf Python-3.12.15.tgz && cd Python-3.12.15
    ./configure --enable-optimizations --with-ensurepip=install
    make -j"$(nproc)" && sudo make altinstall      # altinstall: adds python3.12, leaves the system python3 alone
    python3.12 -c "import tkinter, ssl, sqlite3; print('ok')"

## 2. Make the Descriptron environment (once)

Open a terminal: **Windows** - Start menu, type `cmd`, open "Command Prompt". **macOS** - Terminal.

**Windows**

    py -3.12 -m venv %USERPROFILE%\descriptron-env
    %USERPROFILE%\descriptron-env\Scripts\activate

Python 3.12 does **not** have to be on the PATH: `py -3.12` finds it by version, and `activate` puts the
environment's own Python, `pip` and `descriptron-gui` first on the PATH - for that window only, so run the activate
line again in every new Command Prompt. Use Command Prompt (`cmd`), not PowerShell (there the script is
`Scripts\Activate.ps1` and is often blocked; `Set-ExecutionPolicy -Scope CurrentUser RemoteSigned` allows it).
Optional desktop shortcut - a text file `Descriptron.bat` containing:

    call %USERPROFILE%\descriptron-env\Scripts\activate
    descriptron-gui

**macOS / Linux**

    python3.12 -m venv ~/descriptron-env
    source ~/descriptron-env/bin/activate

The prompt now starts with `(descriptron-env)`. Check with `python --version`: it must say 3.12.x (if not, the
environment was made with another Python - delete the folder and repeat with `py -3.12` / `python3.12`). **Every command below must be typed in a terminal where the
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
the terminal shows the progress, and the GUI opens when it is done. On Windows the question window can open **behind**
the Command Prompt, which then seems to hang: the terminal says so (from 2.8.1) - look for the window on the taskbar. The file is saved here and found
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

## Updating to a new version

Activate the environment as above, then:

    pip install --upgrade descriptron

This updates the GUI and the analysis programs together (their versions always match). If you installed SAM 3,
update it separately: `pipx upgrade descriptron-sam3`. Your annotations and settings are not touched.

Older GUI versions stay installed. `descriptron-gui --which` prints which GUI file starts and the package version;
to start an older one, set `DESCRIPTRON_GUI_VERSION` (e.g. `set DESCRIPTRON_GUI_VERSION=v94` on Windows,
`DESCRIPTRON_GUI_VERSION=v94 descriptron-gui` on macOS / Linux).

## Optional: SAM 3 (find every seta at once)

SAM 3 lives in its own environment. With the Descriptron environment **not** active:

    pip install pipx
    pipx ensurepath                         (then open a new terminal)
    pipx install --python python3.12 descriptron-sam3

Request access at https://huggingface.co/facebook/sam3, create a read token in your Hugging Face settings, and log
in once: `pipx run --spec huggingface_hub hf auth login`. The GUI's dark green **Find all instances (SAM 3)**
button then finds it. Step by step (account, access, token, Windows/macOS/Linux), also for the DINOv3 models used by
DINOMask and DINOLand: [HUGGINGFACE_TOKEN.md](HUGGINGFACE_TOKEN.md).

## If something goes wrong

| message | fix |
|---|---|
| `SAM2 is not installed` / `No module named 'sam2'` | step 4 was not run in the active environment: activate it (step 2) and repeat step 4 |
| `git` is not recognised / `Cannot find command 'git'` | use the zip line in step 4 (no Git needed), or put Git on the PATH: [INSTALL_GIT.md](INSTALL_GIT.md) |
| `py` / `python3.12` not found | Python 3.12 is not installed, or (Windows) "Add python.exe to PATH" was not ticked: reinstall Python |
| downloaded `Python-3.12.x.tgz` and there is no installer in it | that file is source code; on Windows / macOS use the 3.12.10 installer (step 1) |
| the Python Install Manager installed 3.13 / 3.14 / 3.15 instead of 3.12 | `py install 3.12`, then make the environment with `py -3.12` (step 2); the newer Python can stay |
| `No matching distribution found for torch` / `torchvision` | the environment uses a Python that is too new: check `python --version` in it, remake it with Python 3.12 (step 2) |
| `tkinter is missing` (Linux) | `sudo apt install python3-tk` |
| `descriptron-gui` not found | the environment is not active (step 2) |
| the GUI is slow | no NVIDIA card or the CPU PyTorch was installed: repeat step 3 with the NVIDIA command |
