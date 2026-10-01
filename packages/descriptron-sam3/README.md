# descriptron-sam3

SAM 3 for [Descriptron](https://github.com/alexrvandam/Descriptron): outline one or a few examples of a small
structure (for example setae on a wing or genitalia), click on them, or type a word, and SAM 3 proposes every
matching instance as a COCO file. The Descriptron GUI (descriptron-gui 2.5.1 and later) runs it from the dark
green **Find all instances (SAM 3)** button and shows the instances together, one colour each, to be labelled in
one step.

SAM 3 needs Python 3.12 and PyTorch >= 2.7, so install it in its own environment. pipx does that and puts the
`descriptron-sam3` command on your PATH, where the GUI finds it:

    pipx install --python python3.12 descriptron-sam3

The SAM 3 checkpoints are gated: request access at https://huggingface.co/facebook/sam3 (SAM License), then log in
once (`pipx run --spec huggingface_hub hf auth login`, or `hf auth login` in the environment). The first run
downloads the checkpoint (3.3 GB). A CUDA GPU is strongly recommended.

    descriptron-sam3 --images wings/ --text bristle --category seta --out setae.json
    descriptron-sam3 --images wing1.tif --boxes "1200,800,1260,1100" --out setae.json
    descriptron-sam3 --images wing1.tif --points "1520,4040,1" --points_mode each --tile 0 --out one_seta.json

Results are proposals to review in the GUI, not finished annotations. Examples work within the image they are
drawn on; a word searches every image tile by tile. If the GUI cannot find the command, point it at the Python of
the environment that has descriptron-sam3: `DESCRIPTRON_SAM3_PYTHON=/path/to/python`.

Licence: Apache-2.0 (see NOTICE for the CLIP vocabulary file and for SAM 3's own licence).
