import os
import sys
import cv2

# 현재 스크립트 기준 상대경로로 pyopenpose 불러오기
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.join(CURRENT_DIR, "openpose_wrapper"))

from openpose import pyopenpose as op

params = {
    "model_folder": "./models",
    "model_pose": "BODY_25",
    "net_resolution": "-1x256",
    "disable_multi_thread": True
}

opWrapper = op.WrapperPython()
opWrapper.configure(params)
opWrapper.start()

cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    datum = op.Datum()
    datum.cvInputData = frame

    datums_ptr = op.VectorDatum()
    datums_ptr.append(datum)

    opWrapper.emplaceAndPop(datums_ptr)

    cv2.imshow("OpenPose", datum.cvOutputData)

    if cv2.waitKey(1) == 27:
        break

cap.release()
cv2.destroyAllWindows()
