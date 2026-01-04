# QMAP UDP stream format (TIMF2-style)

This describes how the UDP IQ stream **to QMAP** is expected to look.  
The bridge script generates a **Linrad TIMF2**-compatible packet format.

---

## Transport

- **UDP**, continuous stream of fixed-size packets
- Typical rate at **96 kS/s** with **174 samples per packet**:
  - `96 000 / 174 ≈ 552 packets/second`
- **Little-endian** encoding for all header fields and sample data.

---

## Packet size and structure

Each UDP datagram is **1416 bytes**:

- **24-byte TIMF2 header**
- **1392-byte payload** = `174 sample-slots × 8 bytes`

So: `24 + 174*8 = 1416 bytes`.


+-------------------+------------------------------+
| 24-byte header    | 174 sample-slots × 8 bytes   |
+-------------------+------------------------------+


---

##  TIMF2 header (24 bytes)

Packed as:
(little-endian)

| Field       | Type                 | Bytes | Meaning |
|------------|----------------------|------:|---------|
| `cfreq_mhz`| float64 (`d`)        | 8     | **Center frequency in MHz** (RF center shown by QMAP) |
| `msec`     | int32 (`i`)          | 4     | Milliseconds since **UTC midnight** (timestamp) |
| `userfreq` | float32 (`f`)        | 4     | User/offset frequency (often 0; may be used for display) |
| `iptr`     | int32 (`i`)          | 4     | Sample pointer / running index (**should be monotonic**) |
| `iblk`     | uint16 (`H`)         | 2     | Block counter (increments per packet) |
| `nrx`      | int8 (`b`)           | 1     | Receiver number / stream ID (often -1 or 0 depending on receiver) |
| `iusb`     | uint8 (`c`)          | 1     | Mode/marker byte |

### Continuity expectations

Receivers commonly use **`iptr` and/or `iblk`** to detect gaps.  
Best practice is:

- `iptr` increases by **174** every packet (one increment per sample-slot)
- `iblk` increments by **1** every packet (wrap is allowed, but some receivers dislike it)

---

##  Payload (174 sample-slots × 8 bytes)

TIMF2 calls them “doubles” because each slot is **8 bytes**, regardless of whether the IQ is float or int16.

###  Float32 IQ mode (recommended / direct)

Each sample-slot is:


** f  f**


- 4 bytes float32 = **I**
- 4 bytes float32 = **Q**

Typical range:

- **-1.0 … +1.0** (normalized full-scale)

---

###  Int16-in-8-byte-slot mode (compatibility)

Each sample-slot is 8 bytes, but only the first 4 contain IQ:

** h  h  0 0 0 0**

- 2 bytes int16 = **I**
- 2 bytes int16 = **Q**
- 4 bytes padding = **zeros**

Scaling:

- The bridge can apply an `int16_scale` so that normalized amplitudes map to an int16 range.

---


