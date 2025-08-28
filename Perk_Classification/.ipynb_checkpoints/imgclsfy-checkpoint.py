
import subprocess

DATA_DIR = './imagecl/data/imagedataset/'
import os

from ultralytics import YOLO


# Load a model
#model = YOLO("./yolov8n-cls.pt")  # load a pretained model

# Use the model
#results = model.train(data=DATA_DIR, epochs=20, imgsz=640)  # train the model

# command_to_run = "!scp -r ./runs './imagecl' "
# subprocess.run(command_to_run, shell=True, check=True)
#runs/classify/train6/weights/best.pt


from ultralytics import YOLO

model = YOLO('./runs/classify/train11/weights/best.pt')




from tkinter import Button
import gradio as gr

def classify_image(inp):
     results = model(inp)
    
     labels_dict = results[0].names
     probs = results[0].probs.data.tolist()
     print(probs)
     
     if(probs[1]>0.75):
        return results[0].names[1]
     elif (probs[1]>0.50):
        return "ambigious"
     return results[0].names[0]

    

iface = gr.Interface(
    fn=classify_image,
    inputs=gr.Image(label="image"),
    outputs="text",
    title="Image Classification",
    description="Upload an image for classification.",
    theme="compact"
)

# Launch the interface
iface.launch(server_port = 7880, share=True)


