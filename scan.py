import json
import numpy as np
import cv2
import time
import logUtils

from os import listdir
from os.path import isfile, join

logFile = logUtils.logOpen("./log")
errorLogFile = logUtils.errorOpen("./log")


## Get confih from json
try:
    with open('config.json', "r") as f:
        #stuff goes here
        data = json.load(f)
      
        f.close()
except BaseException as err:
    #do what you want if there is an error with the file opening
    logUtils.writeLog(errorLogFile, "json error , please check : "+str(err))
    quit()


streamUrl = data['streamLink']
version = data["version"]
logUtils.writeLog(logFile, "Start OCR , version : "+ version)
logUtils.writeLog(logFile, "stream : "+streamUrl)

## Start handel Video
vcap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
fps = max(vcap.get(cv2.CAP_PROP_FPS) % 100, 0) or 30.0

#wt =  1/fps

videoWidth = vcap.get(cv2.CAP_PROP_FRAME_WIDTH)   # float `width`
videoHeight = vcap.get(cv2.CAP_PROP_FRAME_HEIGHT)  # float `height`
print("videoWidth : ", videoWidth)
print("videoHeight : ", videoHeight)

cam_quit = 0  # Loop control variable
i = 0


while True:
    ##logUtils.writeLog(logFile,"CEHCK =================\n")
    start_time = time.time()
    # Capture frame-by-frame
    ret, frame = vcap.read()
    """---------------------------------------------------------------------------------Start handle Image Logic---------------------------------------------------------------------------------"""

 
    # this is the part to add to your code
    b3Start = 10   
   
    b3Position = [b3Start, 200, b3Start+150,200+70]
    b2Position = [b3Position[2]+10, 200, b3Position[2]+10+70, 350]
        
    
    #B3
    cv2.rectangle(frame, (b3Position[0], b3Position[1]),
                  (b3Position[2], b3Position[3]),  (0, 0, 255), 2)
    #b2
    cv2.rectangle(frame, (b2Position[0], b2Position[1]),
                  (b2Position[2], b2Position[3]),  (0, 0, 255), 2)

    cv2.imshow('frame', frame)





    """---------------------------------------------------------------------------------End Image Logic---------------------------------------------------------------------------------"""
    if cv2.waitKey(22) & 0xFF == ord('q'):
        ##cv2.imwrite('output_stream.jpg', frame)
        break

    time.sleep(1/fps)
    ##cam_quit += 1
     ### logUtils.writeLog(logFile,"CEHCK FINISH =================\n")
vcap.release()
cv2.destroyAllWindows()
print("Video stop")


