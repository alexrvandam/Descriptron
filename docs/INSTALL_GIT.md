# Git: installing it and getting it on the PATH (Windows and macOS)

Descriptron needs Git for one thing only: installing the **SAM2 library** from Meta's GitHub
([INSTALL_PIP.md](INSTALL_PIP.md), step 4). **You can skip Git entirely**: pip installs SAM2 just as well from
GitHub's zip download (see "No Git at all" below).

"On the PATH" means a terminal can find the `git` program when you type `git`. If it is not, you see
`'git' is not recognized as an internal or external command` (Windows) or `command not found: git` (macOS), and
`pip install git+https://...` fails with `Cannot find command 'git'`.

---

## No Git at all (simplest)

Install SAM2 from the zip download instead of with `git+`. In the activated Descriptron environment:

**Windows (Command Prompt)**

    set SAM2_BUILD_CUDA=0
    pip install https://github.com/facebookresearch/sam2/archive/refs/heads/main.zip

**macOS / Linux**

    SAM2_BUILD_CUDA=0 pip install https://github.com/facebookresearch/sam2/archive/refs/heads/main.zip

It installs the same package as the `git+` command.

---

## Windows

### Install Git (with the PATH option)

1. Download "Git for Windows" from https://git-scm.com/downloads/win (the **64-bit** installer on any recent PC)
   and run it.
2. Click **Next** through the screens, keeping the defaults. On the screen titled **"Adjusting your PATH
   environment"**, make sure the middle option is selected: **"Git from the command line and also from 3rd-party
   software"** (the default). This is the option that puts Git on the PATH.
3. Finish the installer.
4. **Close every Command Prompt / PowerShell window and open a new one.** A window that was already open does not
   see the new PATH.
5. Check: `git --version` should print something like `git version 2.51.0.windows.1`.
6. Activate the Descriptron environment again before using pip:
   `%USERPROFILE%\descriptron-env\Scripts\activate`

Alternative, if the computer has `winget` (Windows 10/11): in a new Command Prompt,
`winget install --id Git.Git -e --source winget`, then do steps 4-6.

### Git is installed but still "not recognized"

The installer was probably run with a different PATH option. Either run the installer again and choose the option
in step 2, or add Git to the PATH by hand:

1. Press the Windows key, type **environment variables**, open **"Edit the environment variables for your
   account"**.
2. In the top list ("User variables"), select **Path** and click **Edit...**.
3. Click **New** and paste `C:\Program Files\Git\cmd`
   (check that this folder exists; if Git was installed for one user only it may be
   `C:\Users\<your name>\AppData\Local\Programs\Git\cmd`).
4. **OK**, **OK**, then close and reopen Command Prompt, and check with `git --version`.

---

## macOS

### Install Git

macOS installs Git with Apple's command-line developer tools:

1. Open **Terminal** and type `git --version`.
2. If Git is missing, a window offers to install the **command line developer tools**: click **Install** and wait
   (several minutes). Or start it yourself: `xcode-select --install`.
3. Close Terminal, open a new one, and check: `git --version`.

With Homebrew (https://brew.sh) you can instead run `brew install git`.

### Git is installed but still "command not found"

- Open a **new** Terminal window after installing.
- If `xcode-select --install` says the tools are already installed but `git` still fails, reset them:
  `sudo xcode-select --reset`, then open a new Terminal.
- Homebrew's Git lives in `/opt/homebrew/bin` (Apple silicon) or `/usr/local/bin` (Intel). If Homebrew printed
  "Next steps" lines about adding it to your PATH, run those lines once, then open a new Terminal.

---

## Linux

    sudo apt install git          # Debian/Ubuntu
    sudo dnf install git          # Fedora/RHEL
