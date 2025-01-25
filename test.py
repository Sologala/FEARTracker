import cv2

img = cv2.imread('/mnt/HardDisk/datasets/SOT/GOT-10K/train/GOT-10k_Train_002081/00000001.jpg')

bboxs = [] 
with open("/mnt/HardDisk/datasets/SOT/GOT-10K/train/GOT-10k_Train_002081/groundtruth.txt", "r") as f:
    bboxs = [list(map(float, l.split(","))) for l in f.readlines()]

bbox = [[int(i) for i in bbx] for bbx in bboxs]
bbx = bbox[0]
bbx[2] += bbx[0]
bbx[3] += bbx[1]
img = cv2.rectangle(img, bbx[:2], bbx[2:], (255, 0, 0))
cv2.imshow("img", img)
cv2.waitKey(0)
