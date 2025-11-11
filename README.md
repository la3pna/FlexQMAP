### Flex radio to QMAP Bridge

------

Usage: 

     python .\qmap_flex.py --radio 192.168.1.2 --daxiq 1 --gain 30 --invertq


Setup: 
- Install Python, make sure to select "add to path"
- Install dependencies, in a terminal type:
    python -m pip install --upgrade numpy sounddevice
- In Windows sound, set DAX IQ RX 1 to "2 channels, 24-bit, 96000Hz"
- 
  <img width="404" height="462" alt="image" src="https://github.com/user-attachments/assets/1c7cf4b1-a1e2-4473-9b8f-3f3e595d6544" />

- In DAX set IQ stream to 96000Hz sample rate:

  <img width="295" height="126" alt="image" src="https://github.com/user-attachments/assets/4dc78171-a2da-4321-b11a-d3f01b2732ce" />

- In SmartSDR select DAX and set that to use the IQ channel 1:
- 
  <img width="414" height="135" alt="image" src="https://github.com/user-attachments/assets/55d1eb6c-b982-4eb1-98fb-11ea405e5963" />

- Run the script in terminal:

      python .\qmap_flex.py --radio 192.168.1.2 --daxiq 1 --gain 30 --invertq
