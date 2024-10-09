import numpy as np
import pandas as pd
import cv2

#load video
video = cv2.VideoCapture('meow/video.mp4')
print(video.isOpened())
x=0
while video.isOpened():
    ret, frame = video.read()
    if ret:
        cv2.imshow(f'frame{x}', frame)
        cv2.imwrite(f'meow/frame{x}.jpg', frame)
        x+=1
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    else:
        break
video.release()
cv2.destroyAllWindows()
#predict the frame in between every 2 frames to double the frame rate
x=0
while x<10:
    frame1 = cv2.imread(f'meow/frame{x}.jpg')
    frame2 = cv2.imread(f'meow/frame{x+1}.jpg')
    frame1 = cv2.resize(frame1, (frame2.shape[1], frame2.shape[0]))
    frame = cv2.addWeighted(frame1, 0.5, frame2, 0.5, 0)
    cv2.imwrite(f'meow/frame{2*x}.jpg', frame)
    x+=1
#combine the frames to make a video
img_array = []
for i in range(20):
    img = cv2.imread(f'meow/frame{i}.jpg')
    height, width, layers = img.shape
    size = (width, height)
    img_array.append(img)
out = cv2.VideoWriter('meow/output.avi', cv2.VideoWriter_fourcc(*'DIVX'), 15, size)
for i in range(len(img_array)):
    out.write(img_array[i])
out.release()

