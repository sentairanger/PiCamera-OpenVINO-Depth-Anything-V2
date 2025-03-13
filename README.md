# PiCamera-OpenVINO-Depth-Anything-V2
This project uses Depth Anything V2 and OpenVINO to perform depth estimation with a Pi Camera. 

## Getting Started

This project was tested with a Raspberry Pi 4 with 2 GB of RAM. However, it's recommended to either overclock the Pi 4 or use a Pi 5. A CM4 and CM5 may also be used as well. Any Pi Camera can be used including third party cameras.

To get started you will need to install OpenVINO and PyTorch on the Host Machine and the Pi. The reason you need to do this is that you will have to convert the PyTorch model to OpenVINO IR which I will explain shortly. To do this it's required to have a virtual environment with the system wide packages. Run these commands to build the environment and then activate it.

* `python3 -m venv --system-site-packages torchenv`
* `source torchenv/bin/activate`
When building the environment on your host machine it's best to not use the system wide packages option and just run this:

* `python3 -m venv torchenv`

After activating the environment next install OpenVINO and PyTorch:

* `pip install openvino`
* `pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu`

Note: if you are using a Pi with a Radeon GPU use the ROCm option. 

Next, you will have to clone the [repo](https://github.com/DepthAnything/Depth-Anything-V2) for Depth Anything. Make sure you do that for both your host machine and the Pi. After that, clone this repo on your host machine first inside the `Depth-Anything-V2` folder. Then since we need our model in IR form we will run the `conversion-model.py` code. The XML model should now reside on the directory. Next use scp to copy the model to your Pi. Before you do make sure you do this inside the `Depth-Anything-V2` folder on your Pi. So you would run `scp depth-anything-vits.* username@ip-address-of-pi:/home/Documents/Depth-Anything-V2/`. Next you should move the `depth_picamera2.py` code to the `Depth-Anything-V2` folder because it won't run properly. After that you can run the code on the Pi and the video should appear. Note that you may get either very low frames or close to 0 FPS. 
