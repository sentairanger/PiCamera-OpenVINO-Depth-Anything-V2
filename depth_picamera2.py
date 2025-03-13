from openvino import Core 
from depth_anything_v2.util.transform import Resize, NormalizeImage, PrepareForNet
from torchvision.transforms import Compose
import cv2 
import numpy as np
import torch
from picamera2 import Picamera2
import time 

def get_depth_map(output, w, h):
    depth = cv2.resize(output, (w, h))

    depth = (depth - depth.min()) / (depth.max() - depth.min()) * 255.0
    depth = depth.astype(np.uint8)

    depth = cv2.applyColorMap(depth, cv2.COLORMAP_INFERNO)

    return depth

model = "depth_anything_v2_vits.xml"
ie = Core()
compiled_model = ie.compile_model(model, device_name="CPU")
transform = Compose(
    [
        Resize(
            width=518,
            height=518,
            resize_target=False,
            ensure_multiple_of=14,
            resize_method="lower_bound",
            image_interpolation_method=cv2.INTER_CUBIC,
        ),
        NormalizeImage(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        PrepareForNet(),
    ]
)

picam2 = Picamera2()
picam2.preview_configuration.main.size = (640, 480)
picam2.preview_configuration.main.format = "RGB888"
picam2.preview_configuration.align()
picam2.configure("preview")
picam2.start()
fps = 0
pos = (30, 60)
font=cv2.FONT_HERSHEY_SIMPLEX
height=1.5
weight=3
myColor=(0,255,0)
while True:
    tStart = time.time()
    image = picam2.capture_array()
    h, w = image.shape[:2]
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB) / 255.0
    image = transform({"image": image})["image"]
    image = torch.from_numpy(image).unsqueeze(0).numpy()
    result = compiled_model(image)[0]
    depth_map = get_depth_map(result[0], w, h)
    cv2.putText(depth_map,str(int(fps))+' FPS',pos,font,height,myColor,weight)
    cv2.imshow("Camera", depth_map)
    if cv2.waitKey(1)==ord('q'):
        break
    tEnd = time.time()
    loopTime = tEnd - tStart
    fps = .9*fps + .1*(1/loopTime)
cv2.destroyAllWindows(
