#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Flex DAX IQ (stereo float32) -> QMAP (Linrad TIMF2-UDP) bridge
"""

import argparse, socket, struct, sys, threading, time, math, select
from datetime import datetime, timezone
from typing import Optional, Dict
import numpy as np
import sounddevice as sd

# ---- QMAP/TIMF2 ----
PKT_DOUBLES = 174
FRAMES_PER_PACKET = PKT_DOUBLES
BYTES_PER_PACKET = 1416
HEADER_FMT = "<d i f i H b c"  # cfreq(MHz), msec, userfreq, iptr, iblk, nrx, iusb

# ---- Radio-frekvens-state (MHz) ----
class RadioFreqState:
    def __init__(self):
        self.lock = threading.Lock()
        self.pan_center_by_handle: Dict[str,float] = {}
        self.pan_for_daxiq: Dict[int,str] = {}
        self.slice_freq_by_dax: Dict[int,float] = {}
        self.last_any_pan_center: Optional[float] = None
    def set_pan_center(self, handle: str, mhz: float):
        with self.lock:
            self.pan_center_by_handle[handle] = mhz
            self.last_any_pan_center = mhz
    def set_daxiq_pan(self, ch: int, handle: str):
        with self.lock:
            self.pan_for_daxiq[ch] = handle
    def set_slice_freq_for_dax(self, ch: int, mhz: float):
        with self.lock:
            self.slice_freq_by_dax[ch] = mhz
    def get_center_mhz_for(self, daxiq_ch: int) -> Optional[float]:
        with self.lock:
            h = self.pan_for_daxiq.get(daxiq_ch)
            if h and h in self.pan_center_by_handle:
                return self.pan_center_by_handle[h]
            if daxiq_ch in self.slice_freq_by_dax:
                return self.slice_freq_by_dax[daxiq_ch]
            return self.last_any_pan_center

# ---- SmartSDR API ----
def api_reader_thread(radio_ip: str, daxiq_ch: int, state: RadioFreqState, debug=False):
    port = 4992
    def strip_prefix(txt: str) -> str:
        if "|" in txt and txt[0] in ("S","R","H","M"):
            return txt.split("|",1)[1]
        return txt
    while True:
        try:
            s = socket.create_connection((radio_ip, port), timeout=5.0)
            s.setblocking(False)
            f = s.makefile("rwb", buffering=0)
            t0 = time.time()
            while time.time() - t0 < 1.0:
                r, _, _ = select.select([s], [], [], 0.05)
                if not r: continue
                line = f.readline()
                if not line: break
                if debug: sys.stderr.write(line.decode(errors="ignore"))
            for c in (b"C1|sub pan all\n", b"C2|sub slice all\n", b"C3|sub daxiq all\n"):
                f.write(c)
            buf = b""; last_msg = time.time()
            while True:
                r, _, _ = select.select([s], [], [], 0.2)
                if r:
                    chunk = s.recv(4096)
                    if not chunk: raise OSError("socket closed")
                    buf += chunk
                    while b"\n" in buf:
                        line, buf = buf.split(b"\n", 1)
                        msg = strip_prefix(line.decode(errors="ignore").strip())
                        if "display pan " in msg:
                            try:
                                parts = msg.split(); handle = parts[2]
                                for tok in parts:
                                    if tok.startswith("center="):
                                        mhz = float(tok.split("=",1)[1])
                                        state.set_pan_center(handle, mhz)
                                        if debug: print(f"[API] pan {handle} center={mhz:.6f} MHz")
                                        break
                            except Exception: pass
                        if "daxiq " in msg:
                            try:
                                parts = msg.split(); ch = int(parts[1])
                                if ch == daxiq_ch:
                                    for tok in parts[2:]:
                                        if tok.startswith("pan="):
                                            handle = tok.split("=",1)[1]
                                            state.set_daxiq_pan(ch, handle)
                                            if debug: print(f"[API] daxiq {ch} -> pan {handle}")
                                            break
                            except Exception: pass
                        if msg.startswith("slice "):
                            try:
                                mhz = None; dax = None
                                for tok in msg.split():
                                    if tok.startswith("freq="): mhz = float(tok.split("=",1)[1])
                                    elif tok.startswith("dax="): dax = int(tok.split("=",1)[1])
                                if mhz is not None and dax is not None:
                                    state.set_slice_freq_for_dax(dax, mhz)
                                    if debug: print(f"[API] slice dax={dax} freq={mhz:.6f} MHz")
                            except Exception: pass
                        last_msg = time.time()
                if time.time() - last_msg > 10:
                    try: f.write(b"C9|version\n"); last_msg = time.time()
                    except Exception: break
            s.close()
        except Exception as e:
            if debug: print(f"[API] reconnect in 2s ({e})")
            time.sleep(2.0)

# ---- Diverse ----
def ms_since_midnight_utc() -> int:
    now = datetime.now(timezone.utc)
    midnight = now.replace(hour=0, minute=0, second=0, microsecond=0)
    return int((now - midnight).total_seconds() * 1000)

def find_dax_device(name_hint: str) -> Optional[int]:
    devices = sd.query_devices(); hostapis = sd.query_hostapis()
    wasapi_index = None
    for i, ha in enumerate(hostapis):
        if "WASAPI" in ha.get("name",""): wasapi_index = i; break
    candidate = None
    for idx, d in enumerate(devices):
        if d["max_input_channels"] >= 2:
            name = d.get("name","")
            if name_hint.lower() in name.lower():
                if wasapi_index is None or d["hostapi"] == wasapi_index:
                    return idx
                candidate = idx
    return candidate

class LinearResampler:
    def __init__(self, in_rate: float, out_rate: float = 96000.0):
        self.in_rate = float(in_rate); self.out_rate = float(out_rate)
        self.inc = self.in_rate / self.out_rate; self.t = 0.0
        self.buf = np.zeros((0,2), dtype=np.float32)
    def push(self, x): self.buf = x.copy() if self.buf.size==0 else np.vstack((self.buf,x))
    def available_output_frames(self) -> int:
        if self.buf.shape[0] < 2: return 0
        max_src_index = self.buf.shape[0] - 1 - 1e-6
        if self.t > max_src_index: return 0
        return int(np.floor((max_src_index - self.t) / self.inc)) + 1
    def pop_frames(self, n_out: int) -> np.ndarray:
        if n_out <= 0: return np.zeros((0,2), np.float32)
        idx = self.t + self.inc * np.arange(n_out, dtype=np.float64)
        i0 = np.floor(idx).astype(np.int64); frac = (idx - i0).astype(np.float32)
        i1 = np.clip(i0 + 1, 0, self.buf.shape[0]-1)
        y = (1.0 - frac)[:,None]*self.buf[i0,:] + frac[:,None]*self.buf[i1,:]
        self.t += self.inc * n_out
        drop = int(np.floor(self.t))
        if drop > 0: self.buf = self.buf[drop:,:]; self.t -= drop
        return y.astype(np.float32)

# ---- Bro ----
def run_bridge(args):
    state = RadioFreqState()
    threading.Thread(target=api_reader_thread, args=(args.radio, args.daxiq, state, args.debug), daemon=True).start()

    name_hint = args.device if args.device else f"DAX IQ RX {args.daxiq}"
    dev_index = find_dax_device(name_hint)
    if dev_index is None:
        print(f"Fant ikke enhet som matcher '{name_hint}'.", file=sys.stderr); sys.exit(2)

    dev_info = sd.query_devices(dev_index)
    default_sr = float(dev_info["default_samplerate"])
    print(f"Bruker input device #{dev_index}: {dev_info['name']}")
    print(f"Sender til {args.host}:{args.port} | DAXIQ={args.daxiq} | venter på pan/slice fra {args.radio}:4992")

    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    target = (args.host, args.port)

    use_int16 = args.int16
    nrx = 1 if use_int16 else args.nrx  # +1 for i*2, ellers -1/+2 for r*4
    iusb = 0; iptr = 0; iblk = 0

    want_rate = 96000
    use_resampler = False

    try: extra = sd.WasapiSettings(exclusive=True)
    except Exception: extra = None

    try:
        stream = sd.InputStream(
            device=dev_index, samplerate=want_rate, channels=2, dtype="float32",
            blocksize=FRAMES_PER_PACKET, extra_settings=extra
        ); stream.start(); in_rate = want_rate
        if args.debug: print(f"Åpnet WASAPI Exclusive @ {want_rate} Hz" if extra else f"Åpnet @ {want_rate} Hz")
    except Exception as e1:
        if args.debug: print(f"WASAPI Exclusive @96k feilet ({e1}); prøver Shared @{default_sr:.0f} Hz")
        try:
            stream = sd.InputStream(
                device=dev_index, samplerate=default_sr, channels=2, dtype="float32",
                blocksize=int(round(FRAMES_PER_PACKET * default_sr / want_rate)) or 0
            ); stream.start(); in_rate = default_sr
            use_resampler = (abs(in_rate - want_rate) > 1)
            if args.debug: print(f"Åpnet Shared @{in_rate:.0f} Hz  (resampler={'ON' if use_resampler else 'OFF'})")
        except Exception as e2:
            print(f"Feil: klarte ikke å åpne InputStream: {e2}", file=sys.stderr); sys.exit(1)

    resamp = LinearResampler(in_rate=in_rate, out_rate=want_rate) if use_resampler else None
    last_stats_t = time.time(); pkts = 0
    last_rms_t = time.time()

    # Oscillatorer for mix og testtone
    mix_phase = 0.0
    mix_inc = (2.0 * math.pi * (args.mix if args.mix else 0.0)) / want_rate
    tone_phase = 0.0
    tone_inc = (2.0 * math.pi * (args.tone if args.tone else 0.0)) / want_rate

    # Enkel AGC (på float før ev. int16-konvertering)
    agc_gain = 1.0
    def dbfs_from_rms(rms: float) -> float:
        return -120.0 if rms <= 1e-12 else 20.0 * math.log10(rms)
    def agc_update(block_rms: float):
        nonlocal agc_gain
        if not args.agc: return
        cur_db = dbfs_from_rms(block_rms * agc_gain)
        err = args.agc_target_dbfs - cur_db
        # enkel 1. ordens sløyfe: juster dB med en fraksjon per sekund
        step_db = max(-6.0, min(6.0, err)) * (args.agc_speed / 10.0)  # dempet
        agc_gain *= 10.0 ** (step_db / 20.0)

    def preprocess_block(block: np.ndarray) -> np.ndarray:
        b = block.astype(np.float32, copy=True)

        # DC-block
        if args.dcblock:
            b[:,0] -= np.mean(b[:,0], dtype=np.float64)
            b[:,1] -= np.mean(b[:,1], dtype=np.float64)

        # Bruk fast gain
        if args.gain and args.gain != 1.0:
            b *= float(args.gain)

        # AGC (etter fast gain, før iq-rotasjoner)
        if args.agc:
            # rask RMS-estimat
            rms_est = float(np.sqrt(np.mean(b*b)))
            agc_update(rms_est)
            b *= agc_gain

        # Swap / invert
        if args.swapiq:
            b = b[:, [1,0]]
        if args.invertq:
            b[:,1] = -b[:,1]

        # Mix (kompleks frekvensforskyvning)
        nonlocal mix_phase
        if args.mix and args.mix != 0.0:
            n = b.shape[0]
            t = mix_phase + mix_inc * np.arange(n, dtype=np.float32)
            c = np.cos(t).astype(np.float32); s = np.sin(t).astype(np.float32)
            I = b[:,0].copy(); Q = b[:,1].copy()
            b[:,0] = I*c - Q*s
            b[:,1] = I*s + Q*c
            mix_phase = float(t[-1] + mix_inc)

        # Testtone (svak, -30 dBfs)
        nonlocal tone_phase
        if args.tone and args.tone > 0:
            n = b.shape[0]
            t = tone_phase + tone_inc * np.arange(n, dtype=np.float32)
            c = np.cos(t).astype(np.float32) * 0.0316
            s = np.sin(t).astype(np.float32) * 0.0316
            I = b[:,0].copy(); Q = b[:,1].copy()
            b[:,0] = I*c - Q*s
            b[:,1] = I*s + Q*c
            tone_phase = float(t[-1] + tone_inc)

        return b

    def send_packet_block(iq_174: np.ndarray, cfreq_mhz: float):
        nonlocal iblk, pkts, last_stats_t
        msec = ms_since_midnight_utc()
        header = struct.pack(
            HEADER_FMT,
            float(cfreq_mhz),
            int(msec),
            float(args.userfreq),
            int(iptr),
            int(iblk & 0xFFFF),
            int(nrx),
            bytes([iusb])
        )

        if use_int16:
            # Skaler til valgt fullskala (default 8192) – *ikke* 32767
            s = np.clip(iq_174, -1.0, 1.0) * float(args.int16_scale)
            s = s.astype(np.int16)
            payload = bytearray(PKT_DOUBLES * 8); off = 0
            for i in range(PKT_DOUBLES):
                payload[off:off+4] = struct.pack("<hh", int(s[i,0]), int(s[i,1]))
                payload[off+4:off+8] = b"\x00\x00\x00\x00"
                off += 8
        else:
            # r*4 float32
            payload = bytearray(PKT_DOUBLES * 8); off = 0
            for i in range(PKT_DOUBLES):
                payload[off:off+8] = struct.pack("<ff", float(iq_174[i,0]), float(iq_174[i,1]))
                off += 8

        pkt = header + payload
        if len(pkt) == BYTES_PER_PACKET:
            try: sock.sendto(pkt, target); pkts += 1
            except Exception as e: print(f"UDP send error: {e}", file=sys.stderr)

        iblk = (iblk + 1) & 0xFFFF
        if args.debug and (time.time() - last_stats_t) > 1.0:
            lvl_dbfs = 20.0*math.log10(max(1e-9, float(np.sqrt(np.mean(iq_174*iq_174)))))
            print(f"pkts/s={pkts}  cfreq={cfreq_mhz:.6f} MHz  in_rate={in_rate:.0f}  nrx={nrx}  "
                  f"gain={args.gain}  int16_scale={args.int16_scale if use_int16 else 0}  agc={'ON' if args.agc else 'OFF'}  lvl~{lvl_dbfs:.1f} dBFS")
            pkts = 0; last_stats_t = time.time()

    last_rms_t = time.time()

    try:
        with stream:
            while True:
                data, _ = stream.read(stream.blocksize)
                if data is None or data.size == 0: continue
                cfreq_mhz = state.get_center_mhz_for(args.daxiq)
                if cfreq_mhz is None: continue

                if resamp:
                    resamp.push(data)
                    while resamp.available_output_frames() >= FRAMES_PER_PACKET:
                        blk = resamp.pop_frames(FRAMES_PER_PACKET)
                        blk = preprocess_block(blk)
                        send_packet_block(blk, cfreq_mhz)
                else:
                    pos = 0; N = data.shape[0]
                    while pos + FRAMES_PER_PACKET <= N:
                        blk = data[pos:pos+FRAMES_PER_PACKET,:]
                        blk = preprocess_block(blk)
                        send_packet_block(blk, cfreq_mhz)
                        pos += FRAMES_PER_PACKET

                if args.rms and (time.time() - last_rms_t) > 1.0:
                    tmp = data
                    rms = float(np.sqrt(np.mean(tmp*tmp))); mx = float(np.max(np.abs(tmp)))
                    print(f"RMS={rms:.4f}  max|x|={mx:.4f}")
                    last_rms_t = time.time()
    except KeyboardInterrupt:
        print("\nAvslutter.")
    except Exception as e:
        print(f"Feil i hovedloop: {e}", file=sys.stderr); sys.exit(1)

def main():
    ap = argparse.ArgumentParser(description="Flex DAX IQ → QMAP UDP (cfreq i MHz, justerbar int16)")
    ap.add_argument("--radio", required=True, help="IP til Flex-radio (SmartSDR TCP API, port 4992)")
    ap.add_argument("--daxiq", type=int, default=1, help="DAX IQ kanal (default 1)")
    ap.add_argument("--host", default="127.0.0.1", help="QMAP IP/host (default 127.0.0.1)")
    ap.add_argument("--port", type=int, default=50004, help="QMAP UDP-port (default 50004)")
    ap.add_argument("--userfreq", type=float, default=0.0, help="Valgfri userfreq-felt (float)")
    ap.add_argument("--device", default=None, help="Overstyr navn/substring på inputenhet")
    ap.add_argument("--nrx", type=int, default=-1, choices=[-1,2], help="r*4 formatkode: -1 (default) eller +2")
    ap.add_argument("--int16", action="store_true", help="Bruk i*2 (int16) med nrx=+1")
    ap.add_argument("--int16_scale", type=int, default=8192, help="Fullskala for int16 (default 8192)")
    ap.add_argument("--swapiq", action="store_true", help="Bytt (I,Q) -> (Q,I)")
    ap.add_argument("--invertq", action="store_true", help="Sett Q := -Q")
    ap.add_argument("--gain", type=float, default=1.0, help="Skaler inngang (default 1.0)")
    ap.add_argument("--tone", type=float, default=0.0, help="Injiser kompleks testtone i Hz (0=av)")
    ap.add_argument("--dcblock", action="store_true", help="Fjern DC pr. blokk")
    ap.add_argument("--mix", type=float, default=0.0, help="Kompleks frekvensforskyvning i Hz (0=av)")
    ap.add_argument("--agc", action="store_true", help="Aktiver enkel AGC")
    ap.add_argument("--agc_target_dbfs", type=float, default=-20.0, help="AGC mål (dBFS), default -20")
    ap.add_argument("--agc_speed", type=float, default=1.5, help="AGC hastighet (relativ), default 1.5")
    ap.add_argument("--rms", action="store_true", help="Logg RMS/max|x| hvert sekund")
    ap.add_argument("--debug", action="store_true", help="Debugutskrifter")
    args = ap.parse_args()
    run_bridge(args)

if __name__ == "__main__":
    main()
