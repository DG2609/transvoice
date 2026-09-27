"""Always-on-top subtitle window: what was heard (small) and its translation (large).

Drag to move. Right-click (Ctrl-click on macOS) for the menu.
Windows only: Ctrl+Alt+L toggles click-through, Ctrl+Alt+Q quits.
"""
import ctypes
import queue
import threading
import tkinter as tk
from collections import OrderedDict

from .osutil import IS_MAC, IS_WIN

user32 = ctypes.windll.user32 if IS_WIN else None
GWL_EXSTYLE = -20
WS_EX_LAYERED, WS_EX_TRANSPARENT, WS_EX_TOOLWINDOW, WS_EX_NOACTIVATE = 0x80000, 0x20, 0x80, 0x08000000
MOD_ALT, MOD_CONTROL, MOD_NOREPEAT = 0x1, 0x2, 0x4000
WM_HOTKEY, WM_QUIT = 0x0312, 0x0012

BG, FG, FG_DRAFT, FG_DIM, FG_WAIT, FG_ERR = "#101418", "#ffffff", "#b8c2cc", "#9aa4ad", "#6d7780", "#ff8a80"
if IS_WIN:
    UI_FONT, FONTS = "Segoe UI", {"ja": "Yu Gothic UI", "en": "Segoe UI", "vi": "Segoe UI"}
elif IS_MAC:
    UI_FONT, FONTS = "Helvetica Neue", {"ja": "Hiragino Sans", "en": "Helvetica Neue", "vi": "Helvetica Neue"}
else:
    UI_FONT, FONTS = "Noto Sans", {"ja": "Noto Sans CJK JP", "en": "Noto Sans", "vi": "Noto Sans"}
WHO = {"them": "Họ", "me": "Tôi"}


class GlobalHotkeys(threading.Thread):
    """RegisterHotKey needs a thread with its own message loop."""

    def __init__(self, bindings: dict[str, tuple[int, int]], on_hotkey):
        super().__init__(daemon=True)
        self.bindings, self.on_hotkey = bindings, on_hotkey

    def run(self) -> None:
        from ctypes import wintypes

        names = list(self.bindings)
        for i, name in enumerate(names, 1):
            mods, vk = self.bindings[name]
            user32.RegisterHotKey(None, i, mods | MOD_NOREPEAT, vk)
        msg = wintypes.MSG()
        while user32.GetMessageW(ctypes.byref(msg), None, 0, 0) > 0:
            if msg.message == WM_HOTKEY and 1 <= msg.wParam <= len(names):
                self.on_hotkey(names[msg.wParam - 1])
        for i in range(1, len(names) + 1):
            user32.UnregisterHotKey(None, i)

    def stop(self) -> None:
        if self.native_id:
            user32.PostThreadMessageW(self.native_id, WM_QUIT, 0, 0)


class Overlay:
    MAX_ITEMS = 3

    def __init__(self, ui_q: queue.Queue, on_close):
        self.ui_q, self.on_close = ui_q, on_close
        self.size = 18
        self.locked = False
        self.items: OrderedDict[int, tuple] = OrderedDict()

        self.root = root = tk.Tk()
        root.title("TransVoice")
        root.overrideredirect(True)
        root.attributes("-topmost", True)
        root.attributes("-alpha", 0.94)
        root.configure(bg=BG)
        sw, sh = root.winfo_screenwidth(), root.winfo_screenheight()
        self.width = int(sw * 0.6)
        self.left, self.bottom = (sw - self.width) // 2, sh - 90  # the window grows upwards from here
        self.max_height = sh // 2
        root.geometry(f"{self.width}x60+{self.left}+{self.bottom - 60}")

        self.status = tk.Label(root, text="Đang khởi động…", fg=FG_DIM, bg=BG, font=(UI_FONT, 9), anchor="w")
        self.status.pack(fill="x", padx=12, pady=(6, 0))
        self.body = tk.Frame(root, bg=BG)
        self.body.pack(fill="both", expand=True, padx=12, pady=(0, 8))

        self.menu = tk.Menu(root, tearoff=0)
        if IS_WIN:
            self.menu.add_command(label="Khoá: cho click xuyên qua (Ctrl+Alt+L)", command=self.toggle_lock)
        self.menu.add_command(label="Chữ to hơn", command=lambda: self.resize_font(+2))
        self.menu.add_command(label="Chữ nhỏ hơn", command=lambda: self.resize_font(-2))
        self.menu.add_separator()
        self.menu.add_command(label="Thoát (Ctrl+Alt+Q)" if IS_WIN else "Thoát", command=self.close)
        for w in (root, self.body, self.status):
            self._bind_mouse(w)

        root.update_idletasks()
        self.hotkeys = None
        if IS_WIN:  # click-through and global hotkeys use Win32
            self._set_exstyle(add=WS_EX_TOOLWINDOW | WS_EX_NOACTIVATE)
            self.hotkeys = GlobalHotkeys({"lock": (MOD_CONTROL | MOD_ALT, ord("L")),
                                          "quit": (MOD_CONTROL | MOD_ALT, ord("Q"))},
                                         lambda name: self.ui_q.put(("hotkey", name)))
            self.hotkeys.start()
        root.after(50, self._poll)

    # ---- window behaviour ----
    def _hwnd(self) -> int:
        return user32.GetParent(self.root.winfo_id())

    def _set_exstyle(self, add: int = 0, remove: int = 0) -> None:
        hwnd = self._hwnd()
        style = user32.GetWindowLongW(hwnd, GWL_EXSTYLE)
        user32.SetWindowLongW(hwnd, GWL_EXSTYLE, (style | add | WS_EX_LAYERED) & ~remove)

    def toggle_lock(self) -> None:
        self.locked = not self.locked
        if self.locked:
            self._set_exstyle(add=WS_EX_TRANSPARENT)
        else:
            self._set_exstyle(remove=WS_EX_TRANSPARENT)
        self._show_status(self._status_text)

    def resize_font(self, delta: int) -> None:
        self.size = max(10, min(40, self.size + delta))
        for _, (_, src, tr, lang) in self.items.items():
            tr.configure(font=(FONTS.get(lang, UI_FONT), self.size, "bold"))
        self._fit()

    def _bind_mouse(self, w) -> None:
        w.bind("<ButtonPress-1>", self._drag_start)
        w.bind("<B1-Motion>", self._drag_move)
        popup = lambda e: self.menu.tk_popup(e.x_root, e.y_root)  # noqa: E731
        w.bind("<Button-3>", popup)
        if IS_MAC:  # secondary click is Button-2 on macOS Tk, or Ctrl-click
            w.bind("<Button-2>", popup)
            w.bind("<Control-Button-1>", popup)

    def _drag_start(self, e) -> None:
        self._drag = (e.x_root - self.root.winfo_x(), e.y_root - self.root.winfo_y())

    def _drag_move(self, e) -> None:
        dx, dy = self._drag
        self.left, top = e.x_root - dx, e.y_root - dy
        self.bottom = top + self.root.winfo_height()
        self.root.geometry(f"+{self.left}+{top}")

    def _fit(self) -> None:
        """Size the window to its content, keeping the bottom edge where the user put it."""
        self.root.update_idletasks()
        while len(self.items) > 1 and self.root.winfo_reqheight() > self.max_height:
            self.items.popitem(last=False)[1][0].destroy()
            self.root.update_idletasks()
        h = self.root.winfo_reqheight()
        self.root.geometry(f"{self.width}x{h}+{self.left}+{self.bottom - h}")

    # ---- content ----
    _status_text = ""

    def _show_status(self, text: str) -> None:
        self._status_text = text
        lock = "🔒 click xuyên qua · Ctrl+Alt+L để mở" if self.locked else "Kéo để di chuyển · chuột phải: menu"
        self.status.configure(text=f"{text}    ·    {lock}")

    def _upsert(self, u) -> None:
        if u.id not in self.items:
            frame = tk.Frame(self.body, bg=BG)
            frame.pack(fill="x", anchor="w", pady=(4, 0))
            src = tk.Label(frame, bg=BG, fg=FG_DIM, anchor="w", justify="left", wraplength=self.width - 40,
                           font=(FONTS.get(u.lang, UI_FONT), 10))
            src.pack(fill="x")
            tr = tk.Label(frame, bg=BG, fg=FG, anchor="w", justify="left", wraplength=self.width - 40,
                          font=(FONTS.get(u.target, UI_FONT), self.size, "bold"))
            tr.pack(fill="x")
            for w in (frame, src, tr):
                self._bind_mouse(w)
            self.items[u.id] = (frame, src, tr, u.target)
            while len(self.items) > self.MAX_ITEMS:
                old = self.items.popitem(last=False)[1]
                old[0].destroy()
        _, src, tr, _ = self.items[u.id]
        src.configure(text=f"{WHO[u.channel]} · {u.lang.upper()}   {u.text}")
        # Drafts (sentence still growing, or translated from fewer chunks than heard) are grey; final is white.
        if u.target == u.lang:
            tr.configure(text=u.text, fg=FG if u.closed else FG_DRAFT)
        elif u.partial:  # translation still streaming in: words appear as they are generated
            tr.configure(text=u.partial + " …", fg=FG if u.closed else FG_DRAFT)
        elif u.translation:
            tr.configure(text=u.translation + ("" if u.final else " …"), fg=FG if u.final else FG_DRAFT)
        else:
            tr.configure(text="…", fg=FG_WAIT)
        self._fit()

    def _poll(self) -> None:
        try:
            while True:
                kind, payload = self.ui_q.get_nowait()
                if kind == "status":
                    self._show_status(payload)
                elif kind in ("heard", "partial", "translated"):
                    self._upsert(payload)
                    t = payload.timings
                    if kind == "translated" and "lag_ms" in t:
                        self._show_status(f"{payload.lang.upper()}→{payload.target.upper()}  "
                                          f"trễ {t['lag_ms'] / 1000:.1f}s  ·  hàng chờ {t.get('backlog', 0)}")
                elif kind == "error":
                    self.status.configure(text=f"Lỗi: {payload}", fg=FG_ERR)
                elif kind == "hotkey":
                    if payload == "lock":
                        self.toggle_lock()
                    elif payload == "quit":
                        self.close()
                        return
        except queue.Empty:
            pass
        self.root.after(50, self._poll)

    def run(self) -> None:
        self.root.mainloop()

    def close(self) -> None:
        if self.hotkeys is not None:
            self.hotkeys.stop()
        self.on_close()
        self.root.destroy()
