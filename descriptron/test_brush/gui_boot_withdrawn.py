"""Boot a Descriptron GUI file with every window WITHDRAWN (nothing appears on screen) and return its globals.

Used by the v98 point-type GUI tests: the real GUI code runs (widgets, canvas, wrappers, saves) but Tk never maps a
window, dialogs are answered by stubs and mainloop() returns at once. Needs a display connection (Tk), not a visible
window. Run nothing here directly."""
import os
import sys
import tkinter
from tkinter import filedialog, messagebox, simpledialog

ANSWERS = {"asksaveasfilename": None, "askopenfilename": None, "askdirectory": None, "askstring": None,
           "askyesno": True, "askyesnocancel": False, "askokcancel": True, "askinteger": None, "askfloat": None}
MESSAGES = []


def _patch():
    _tk_init, _top_init = tkinter.Tk.__init__, tkinter.Toplevel.__init__

    def tk_init(self, *a, **k):
        _tk_init(self, *a, **k)
        self.withdraw()

    def top_init(self, *a, **k):
        _top_init(self, *a, **k)
        self.withdraw()

    tkinter.Tk.__init__ = tk_init
    tkinter.Toplevel.__init__ = top_init
    tkinter.Wm.deiconify = tkinter.Wm.wm_deiconify = lambda self: None
    tkinter.Misc.mainloop = lambda self, n=0: None
    tkinter.Misc.wait_window = lambda self, window=None: None
    tkinter.Misc.wait_visibility = lambda self, window=None: None
    tkinter.Misc.grab_set = lambda self: None
    tkinter.Misc.grab_set_global = lambda self: None
    for mod in (filedialog, messagebox, simpledialog):
        for name in list(ANSWERS):
            if hasattr(mod, name):
                setattr(mod, name, (lambda n: (lambda *a, **k: ANSWERS[n]))(name))
    for name in ("showinfo", "showerror", "showwarning"):
        setattr(messagebox, name, (lambda n: (lambda *a, **k: MESSAGES.append((n, a, k))))(name))


def boot(gui_path, argv_extra=()):
    gui_path = os.path.abspath(gui_path)
    gui_dir = os.path.dirname(gui_path)
    _patch()
    if gui_dir not in sys.path:
        sys.path.insert(0, gui_dir)
    os.chdir(gui_dir)
    sys.argv = [gui_path] + list(argv_extra)
    g = {"__name__": "__main__", "__file__": gui_path, "__builtins__": __builtins__}
    with open(gui_path) as fh:
        code = compile(fh.read(), gui_path, "exec")
    exec(code, g)
    return g
