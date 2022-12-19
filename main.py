from flask import Flask, redirect, request, render_template, url_for
from flask_restful import Api, Resource
import cv2
import torch
import pandas

app = Flask(__name__)
api = Api(app)

app.config["SECRET_KEY"] = "iitm hackathon"

@app.route('/')
def index():

    if (request.method == "GET"):
        model = torch.hub.load('ultralytics/yolov5', 'custom', path='yolov5/runs/train/exp8/weights/last.pt', force_reload=True)
        cap = cv2.VideoCapture("FullSizeRender.MOV")
        l = []
        l1 = []

        try:
            while True:

                ret, frame = cap.read()

                if ret is False:
                    break
                
                l1.append(cap.get(cv2.CAP_PROP_POS_MSEC))
                l.append(model(cv2.GaussianBlur(frame, (0, 0), 10)).pandas().xyxy[0].name[0])
                print(l)

        except:
            return render_template("index.html", output1=l1, output=l, length=len(l))
    return render_template("index.html", output1=l1, output=l, length=len(l))

if __name__ == "__main__":
    app.run(debug=True)