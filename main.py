import cv2

# img = cv2.imread("person.png")

option = input('''
Enter Camera Number: \n
''')

cap = cv2.VideoCapture(int(option))
cap.set(3, 640)
cap.set(4, 480)

classNames = []
classFile='names.txt'

with open(classFile, 'rt') as file:
    classNames = file.read().rstrip("\n").split('\n')

configPath = 'ssd_mobilenet_v3_large_coco_2020_01_14.pbtxt'
weightsPath = 'frozen_inference_graph.pb'

net = cv2.dnn_DetectionModel(weightsPath, configPath)
net.setInputSize(320, 320)
net.setInputScale(1.0/127.5)
net.setInputMean((127.5, 127.5, 127.5))
net.setInputSwapRB(True)

while True:
    sucess, img = cap.read()

    classIds, confs, bbox = net.detect(img)
    print(classIds, bbox)

    if(len(classIds) != 0):
        for classIds, confidence, box in zip(classIds.flatten(), confs.flatten(), bbox):
            cv2.rectangle(img, box, color=(0, 255, 0), thickness = 3)
            cv2.putText(img, classNames[classIds-1], (box[0]+10, box[1]+30), cv2.FONT_HERSHEY_PLAIN, 2, (255, 0, 0), 2)

        cv2.imshow("Output", img)
        cv2.waitKey(1)