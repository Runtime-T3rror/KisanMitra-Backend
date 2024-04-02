import cv2
from django.http import JsonResponse
from django.views.decorators.csrf import csrf_exempt
import numpy as np
from ultralytics import YOLO
import json
from KisanMitra.settings import BASE_DIR
from os import path

weed_detection_model = YOLO(path.join(BASE_DIR, 'yolo_models/weed.pt'))
disease_detection_model = YOLO(path.join(BASE_DIR, 'yolo_models/disease.pt'))
ripeness_detection_model = YOLO(path.join(BASE_DIR, 'yolo_models/ripeness.pt'))


@csrf_exempt
def detect_weed(req):
    if req.method == 'POST' and req.FILES.get('image'):
        image_file = req.FILES['image'].read()
        nparr = np.frombuffer(image_file, np.uint8)
        img_np = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        print(weed_detection_model.predict(img_np)[0].tojson())
        return JsonResponse({'message': 'Image uploaded successfully'})


@csrf_exempt
def detect_maturity(req):
    if req.method == 'POST' and req.FILES.get('image'):
        image_file = req.FILES['image'].read()
        nparr = np.frombuffer(image_file, np.uint8)
        img_np = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        return JsonResponse(
            {
                'result': json.loads(ripeness_detection_model.predict(img_np)[0].tojson())
            }
        )


@csrf_exempt
def detect_diesease(req):
    if req.method == 'POST' and req.FILES.get('image'):
        image_file = req.FILES['image'].read()
        nparr = np.frombuffer(image_file, np.uint8)
        img_np = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        return JsonResponse({
            'result': json.loads(disease_detection_model.predict(img_np)[0].tojson())
        })
