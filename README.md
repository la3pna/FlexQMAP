### Flex radio to QMAP Bridge

------

Usage: 

     python .\qmap_flex.py --radio 192.168.1.2 --daxiq 1 --gain 30 
     python .\qmap_flex.py --gui


Setup: 
- Install Python, make sure to select "add to path"
- Install dependencies, in a terminal type:
    python -m pip install --upgrade numpy sounddevice
- In Windows sound, set the DAX IQ input to 2 channels at 96000Hz. The device is named `DAX IQ RX 1` with DAXv1 (SmartSDR 4.1.5) and `DAX IQ 1 (FlexRadio DAX)` with DAXv2 (SmartSDR 4.2.18 and later); Windows may show 24-bit or 32-bit depending on the driver version.
- 
  <img width="404" height="462" alt="image" src="https://github.com/user-attachments/assets/1c7cf4b1-a1e2-4473-9b8f-3f3e595d6544" />

- In DAX set IQ stream to 96000Hz sample rate:

  <img width="295" height="126" alt="image" src="https://github.com/user-attachments/assets/4dc78171-a2da-4321-b11a-d3f01b2732ce" />

- In SmartSDR select DAX and set that to use the IQ channel 1:
- 
  <img width="414" height="135" alt="image" src="https://github.com/user-attachments/assets/55d1eb6c-b982-4eb1-98fb-11ea405e5963" />

- set QMAP to:
<img width="402" height="304" alt="image" src="https://github.com/user-attachments/assets/a831d994-3554-4640-b8a7-a984c5225c3c" />

- configure the waterfall so that you get a propper working waterfall. Gain 2, zero 13 works for me. 

- Run the script in terminal:

      python .\qmap_flex.py --radio 192.168.1.2 --daxiq 1 --gain 30 
make sure your radio IP address is correct

<img width="1871" height="535" alt="image" src="https://github.com/user-attachments/assets/20610ab7-97ff-445e-8e95-5f215459582a" />

## Graphical launcher

Run `python .\qmap_flex.py --gui` to open a simple window for the radio address, DAX IQ channel, QMAP destination, gain, mixing, and common IQ/AGC options. The window shows the bridge output and provides Start and Stop buttons. The command-line interface remains available as before.

---

## Some settings
### `--daxiq <n>`
DAX IQ channel number.  

### `--device <substring>`
Optional input-device name override. Without it, the script checks for both `DAX IQ 1 (FlexRadio DAX)` (DAXv2) and `DAX IQ RX 1` (DAXv1). Example: `--device "DAX IQ 1 (FlexRadio DAX)"`.

### `--swapiq` / `--noswapiq`
Swaps I and Q.

### `--invertq`
Inverts Q (Complex conjugate)

### `--gain <linear>`
Linear gain, samples are multiplied by this. 

### `--dcblock`
Removes DC offset from I and Q. Only use if you have a zero spur

### `--agc`
Enables a simple AGC after gain.

### `--agc_target_dbfs <dBFS>`
Target RMS level in dBFS (default: -20).

### `--agc_speed <value>`
How fast the AGC reacts (default: 1.5). Larger = faster.

The AGC values needs to be set together: --agc --agc_target_dbfs -18 --agc_speed 2.0

### `--rms`
Prints basic signal stats.

### `--debug`
Verbose debug output.
