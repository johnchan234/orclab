# orclab

1. Run handleVideoToCardPNG
    - create PNG to ./image/videoCard/

2. 手動整D卡既名, 再放洛個4個FOLDER到 , 再RUN renameFileCard

3. Run handlebgToPck
  - create bg pck to ./pck/
  
3. Run handleCardToPck
    - create card pck to ./pck/
        -all to card inside /videoCard should include to pck

4. run genTrainData.py

5. python train.py --data customdata.yaml --cfg yolov5n.yaml --weights .\1220best.pt --batch 4 --epochs 10

clubs (♣) = 0
diamonds (♦) = 1
hearts (♥) = 2 
spades (♠) = 3


cardToLabel = {
    '0_a': "0",
    '0_2': "1",
    '0_3': "2",
    '0_4': "3",
    '0_5': "4",
    '0_6': "5",
    '0_7': "6",
    '0_8': "7",
    '0_9': "8",
    '0_10': "9",
    '0_j': "10",
    '0_q': "11",
    '0_k': "12",

    '1_a': "13",
    '1_2': "14",
    '1_3': "15",
    '1_4': "16",
    '1_5': "17",
    '1_6': "18",
    '1_7': "19",
    '1_8': "20",
    '1_9': "21",
    '1_10': "22",
    '1_j': "23",
    '1_q': "24",
    '1_k': "25",

    '2_a': "26",
    '2_2': "27",
    '2_3': "28",
    '2_4': "29",
    '2_5': "30",
    '2_6': "31",
    '2_7': "32",
    '2_8': "33",
    '2_9': "34",
    '2_10': "35",
    '2_j': "36",
    '2_q': "37",
    '2_k': "38",

    '3_a': "39",
    '3_2': "40",
    '3_3': "41",
    '3_4': "42",
    '3_5': "43",
    '3_6': "44",
    '3_7': "45",
    '3_8': "46",
    '3_9': "47",
    '3_10': "48",
    '3_j': "49",
    '3_q': "50",
    '3_k': "51",
}

need to run  before run application
	pip install -r requirements.txt
	pip install opencv-python
	pip install websocket-client
	
	pip install sympy
pup instll websocket-cline
	

run script 
	

nvcc --version   

E:\project\pytest\yolov5
py train.py --data customdata.yaml  --weights .\0121_47_best.pt --batch 4 --epochs 30py t

py train.py --data customdata.yaml  --weights .\0130_best.pt --batch 5 --epochs 300

py train.py --data customdata.yaml  --weights .\0130_best.pt --batch 5 --epochs 100

py train.py --data customdata.yaml --cfg yolov5n.yaml --weights .\0118_2best.pt --batch 4 --epochs 30
 py .\customdetect.py --source 0 --weights .\1220best.pt --conf 0.8
 py .\detect.py --source 0 --weights .\0120_3best.pt --conf 0.8
 
 py .\detect.py --source 0 --weights .\0120_3best.pt --conf 0.8
 py .\detect.py --source 0 --weights .\runs\train\exp51\weights\best.pt --conf 0.8

  py .\customdetect2.py --source 0 --weights .\0121_51_best.pt --conf 0.8