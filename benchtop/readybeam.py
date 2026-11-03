"""Run an acquisition under external trigger, then verify READYBeam standby.

Windows only. Drives the installed FISBA Service_RGB v2.00 controls using
Win32 messages. The manufacturer program owns RS-485 and may remain hidden.
"""
import argparse
import ctypes as ct
from ctypes import wintypes as w
from contextlib import contextmanager
from datetime import datetime
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import time

EXE = Path(r"C:\Program Files (x86)\FISBA AG\READYBeam Software v2.00\Service Software\Service_RGB.exe")
APP_SHA = "96487ba9c6456bf7f886a45ec40907e519e9072de846dd8371f354f05bc046ad"
u = ct.WinDLL("user32", use_last_error=True)
k = ct.WinDLL("kernel32", use_last_error=True)
visit = ct.WINFUNCTYPE(w.BOOL, w.HWND, w.LPARAM)
u.SendMessageTimeoutW.argtypes = [w.HWND, w.UINT, w.WPARAM, w.LPARAM, w.UINT, w.UINT, ct.POINTER(ct.c_size_t)]
k.OpenProcess.restype = w.HANDLE
k.QueryFullProcessImageNameW.argtypes = [w.HANDLE, w.DWORD, w.LPWSTR, ct.POINTER(w.DWORD)]
k.CloseHandle.argtypes = [w.HANDLE]
k.CreateMutexW.restype = w.HANDLE
k.CreateMutexW.argtypes = [ct.c_void_p,w.BOOL,w.LPCWSTR]

def send(hwnd, msg, wp=0, lp=0):
    result = ct.c_size_t()
    if not u.SendMessageTimeoutW(hwnd, msg, wp, lp, 2, 2000, ct.byref(result)):
        raise OSError(f"Manufacturer control timed out, message {msg:#x}")
    return result.value

def label(hwnd):
    buf = ct.create_unicode_buffer(256)
    send(hwnd, 13, len(buf), ct.addressof(buf))
    return buf.value

def kind(hwnd):
    buf = ct.create_unicode_buffer(128)
    u.GetClassNameW(hwnd, buf, len(buf))
    return buf.value.split(".")[1:2]

def children(parent):
    found = []
    def collect(h, _):
        if u.GetParent(h) == parent:
            found.append(h)
        return True
    u.EnumChildWindows(parent, visit(collect), 0)
    return found

def one(items):
    items = list(items)
    if len(items) != 1:
        raise RuntimeError(f"Expected one matching manufacturer control, found {len(items)}")
    return items[0]

def named(parent, text):
    return one(h for h in children(parent) if label(h) == text)

def typed(parent, name):
    return one(h for h in children(parent) if kind(h) == [name])

def rect(h):
    r = w.RECT()
    if not u.GetWindowRect(h, ct.byref(r)):
        raise ct.WinError()
    return r

def read_row(group, caption):
    # In v2.00 each read-only status label is right of its caption on the same row.
    at = rect(named(group, caption))
    h = one(h for h in children(group) if kind(h) == ["STATIC"]
            and rect(h).left >= at.right and abs(rect(h).top-at.top) <= 2)
    return label(h)

def forms():
    found = []
    def collect(h, _):
        title = ct.create_unicode_buffer(128)
        u.GetWindowTextW(h, title, len(title))
        if title.value.split() == ["READYBeam", "Software", "V", "2.00"]:
            found.append(h)
        return True
    u.EnumWindows(visit(collect), 0)
    return found

def verify_exe(hwnd):
    pid = w.DWORD()
    u.GetWindowThreadProcessId(hwnd, ct.byref(pid))
    process = k.OpenProcess(0x1000, False, pid.value)
    n = w.DWORD(1024)
    buf = ct.create_unicode_buffer(n.value)
    try:
        if not process or not k.QueryFullProcessImageNameW(process, 0, buf, ct.byref(n)):
            raise ct.WinError()
        if Path(buf.value) != EXE:
            raise RuntimeError("Unexpected executable")
    finally:
        if process:
            k.CloseHandle(process)

def select(group, value):
    h = typed(group, "COMBOBOX")
    text = ct.create_unicode_buffer(value)
    idx = send(h, 0x158, -1, ct.addressof(text))  # CB_FINDSTRINGEXACT
    if idx > 255 or send(h, 0x14e, idx) != idx:
        raise RuntimeError(f"Cannot select {value}")
    send(group, 0x111, (u.GetDlgCtrlID(h) & 65535) | (1 << 16), h)  # CBN_SELCHANGE

class ReadyBeam:
    def __init__(self):
        if hashlib.sha256(EXE.read_bytes()).hexdigest() != APP_SHA:
            raise RuntimeError("Unverified READYBeam application build")
        if not forms():
            info = subprocess.STARTUPINFO()
            info.dwFlags |= subprocess.STARTF_USESHOWWINDOW
            info.wShowWindow = 0
            subprocess.Popen([str(EXE)], cwd=EXE.parent, startupinfo=info)
        end = time.monotonic()+25
        while time.monotonic() < end:
            if forms():
                self.form = one(forms())
                verify_exe(self.form)
                try:
                    named(self.form, "Connected")
                    tab = typed(self.form, "SysTabControl32")
                    self.main = named(tab, "Main")
                    self.state()
                    return
                except RuntimeError:
                    pass
            time.sleep(.3)
        raise RuntimeError("Expected connected READYBeam serial 23, firmware 2.00")

    def state(self):
        named(self.form, "Connected")
        status = named(self.main, "Status")
        serial = read_row(status, "Serial Number [..]")
        if serial != "23" or read_row(status,"Firmware Version [..]") != "2.00":
            raise RuntimeError("Wrong READYBeam identity")
        en, pct = [], []
        for i, ch in enumerate("BGR", 1):
            g = named(self.main, f"CH{i} - {ch}")
            en.append(read_row(g, f"Digital Enable {i}"))
            pct.append(float(read_row(g, f"Power Value {i} [%]")))
        err = named(named(self.main,"Error Status"),"Error Description")
        return dict(time=datetime.now().astimezone().isoformat(),serial=serial,
                    mode=read_row(named(self.main,"Mode"),"Modulation Mode"),
                    enable_BGR=en, percent_BGR=pct, error=label(typed(err,"STATIC")),
                    global_status=read_row(status,"Laser On/Off Status"),
                    module_C=read_row(status,"Laser Module Temperature [°C]"))

    def press(self, name):
        send(named(self.form,name), 0xf5)  # BM_CLICK

    def read(self):
        self.press("Read Config")
        time.sleep(.85)
        return self.state()

    def configure(self, mode):
        before = self.read()
        if mode == "Analog" and before["error"] != "No Error":
            raise RuntimeError(str(before))
        for i,ch in enumerate("BGR",1):
            group = named(self.main,f"CH{i} - {ch}")
            select(group,"Off")
            zero = ct.create_unicode_buffer("0.0")
            if not send(typed(group,"EDIT"),12,0,ct.addressof(zero)):
                raise RuntimeError("Cannot set digital power to zero")
        select(named(self.main,"Mode"),mode)
        self.press("Write Config")
        time.sleep(.9)
        actual = self.read()
        if actual["mode"] != mode or actual["enable_BGR"] != ["Off"]*3 or actual["percent_BGR"] != [0]*3:
            raise RuntimeError(f"READYBeam read-back mismatch: {actual}")
        return actual

@contextmanager
def exclusive():
    h=k.CreateMutexW(None,True,"Local\\FISBA_READYBeam_23_experiment")
    if not h:
        raise ct.WinError()
    try:
        if ct.get_last_error()==183:
            raise RuntimeError("Another READYBeam experiment is active")
        yield
    finally:
        k.CloseHandle(h)

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--log", type=Path)
    p.add_argument("mode",choices=["status","standby","run"])
    p.add_argument("args",nargs=argparse.REMAINDER,help="Python script or -m module and arguments for run")
    args=p.parse_args()
    if (args.mode=="run") != bool(args.args):
        p.error("run requires a Python acquisition; status/standby take no acquisition")
    def record(state):
        line=json.dumps(state)
        print(line,flush=True)
        if args.log:
            with args.log.open("a",encoding="utf-8") as f:
                print(line,file=f)
    with exclusive():
        b=ReadyBeam()
        if args.mode=="status":
            record(b.read())
        elif args.mode=="standby":
            record(b.configure("Digital"))
        else:
            try:
                record(b.configure("Digital"))
                record(b.configure("Analog"))
                subprocess.run([sys.executable,*args.args],check=True)
            finally:
                record(b.configure("Digital"))

if __name__=="__main__":
    main()
