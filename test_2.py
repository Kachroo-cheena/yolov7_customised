import cv2, time
import numpy as np
import pandas as pd
import torch, torchvision
from torchvision import transforms
import os
from utils.datasets import letterbox
from utils.general import non_max_suppression_kpt
from utils.plots import output_to_keypoint,get_keypoints
import matplotlib.pyplot as plt
import glob
from tqdm.auto import tqdm


class YOLO:
    def __init__(
        self, 
        weight_path,
        frame_path
    ):
        self.weight_path = weight_path
        self.frame_path = frame_path
        self.files = glob.glob(os.path.join(self.frame_path, "*.png"))
        self.model = self.load_model()

    def load_model(self):
            device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
            model = torch.load(self.weight_path, map_location=device)['model']
            model.float().eval()
            if torch.cuda.is_available():
                model.half().to(device)
            return model

    def run_inference(self, frame):
        image = cv2.imread(frame)
        print(image)
        image = letterbox(image, 960, stride=64, auto=True)[0]
        image = transforms.ToTensor()(image)
        image = image.unsqueeze(0)
        output, _ = self.model(image)

        output = non_max_suppression_kpt(output, 
                                    0.5, 
                                    0.65, 
                                    nc=self.model.yaml['nc'], 
                                    nkpt=self.model.yaml['nkpt'], 
                                    kpt_label=True)
        with torch.no_grad():
            output = output_to_keypoint(output)
        # print(output.shape)
        self.output = output
        self.image = image
        return self.find_keypoints()
    def find_keypoints(self):
        nimg = self.image[0].permute(1, 2, 0) * 255
        nimg = nimg.cpu().numpy().astype(np.uint8)
        nimg = cv2.cvtColor(nimg, cv2.COLOR_RGB2BGR)
        # print(frame)
        # print(nimg)
        get_keypoints(self.output[0,7:].T,3)
       
    
    def run_model(self): 
        df=[] 
        for i in tqdm(self.files):                  
            tmp = self.run_inference(i)
            tmp["frame"] = i
            df.append(tmp)
        self.df = pd.DataFrame(df)
        

