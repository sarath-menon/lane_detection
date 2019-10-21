from __future__ import division

import cv2

import track
import detect
import argparse
import numpy as np

parser = argparse.ArgumentParser()
parser.add_argument("--path", type=str, help="Video path")

def main(video_path):
    # cap = cv2.VideoCapture(0)
    cap=cv2.VideoCapture('project_video.mp4')

    ticks = 0

    lt = track.LaneTracker(2, 0.1, 500)
    ld = detect.LaneDetector(180)
    while cap.isOpened():
        precTick = ticks
        ticks = cv2.getTickCount()
        dt = (ticks - precTick) / cv2.getTickFrequency()

        ret, frame = cap.read()
        # frame = cv2.flip( frame, -1 )

        # pts1 = np.float32([[0, 0], [640, 0], [640, 352], [0, 352]])
        # pts2 = np.float32([[300, 176], [12, 174], [620, 344], [72, 354]])
        # matrix = cv2.getPerspectiveTransform(pts1,pts2)
        # frame = cv2.warpPerspective(frame, matrix, (500, 500))

        predicted = lt.predict(dt)

        lanes = ld.detect(frame)


        if predicted is not None:
            cv2.line(frame, (predicted[0][0], predicted[0][1]), (predicted[0][2], predicted[0][3]), (0, 0, 255), 7)
            cv2.line(frame, (predicted[1][2], predicted[1][3]), (predicted[1][0], predicted[1][1]), (0, 0, 255), 7)

            center_coordinates_top = ((abs(predicted[1][0]-predicted[0][2]) / 2) + predicted[0][2], predicted[1][1])
            center_coordinates_bottom = ((abs(predicted[0][0]-predicted[1][2]) / 2) + predicted[0][0], predicted[0][1])

            cv2.line(frame, center_coordinates_top, center_coordinates_bottom, (0, 2555, 0), 7)
            cv2.circle(frame, ( int(frame.shape[1]/2), int(frame.shape[0])) , 10, (255, 0, 0) , 2) # Image Center marker

            # cv2.circle(frame, center_coordinates_bottom, 20, (0, 255, 0) , 4) # Center bottom marker
            # cv2.circle(frame, center_coordinates_top, 20, (255, 0, 0) , 4) # Center top marker
            # cv2.circle(frame, (predicted[0][2], predicted[0][3]), 20, (0, 255, 0) , 4) # Left lane marker
            # cv2.circle(frame, (predicted[1][0], predicted[1][1]), 20, (0, 255, 0) , 4) # Right lane marker

            # print('Lane left:',predicted[0][2], predicted[0][3], 'Lane right:',predicted[1][0], predicted[1][1]
            #       ,'Lane center:',center_coordinates_top[0] )

            error = frame.shape[1]/2 - (abs(predicted[0][0]-predicted[1][2]) / 2 + predicted[0][0])
            angle = np.degrees( np.arctan(center_coordinates_top[0] ,center_coordinates_top[1] ))

            a = np.array([0,352])
            b = np.array([center_coordinates_bottom[0], center_coordinates_bottom[1]])
            c = np.array([center_coordinates_top[0],center_coordinates_top[1]])

            ba = a - b
            bc = c - b

            cosine_angle = np.dot(ba, bc) / (np.linalg.norm(ba) * np.linalg.norm(bc))
            angle = 90. - np.degrees(np.arccos(cosine_angle))

            print('Crosstrack Error:',error)
            cv2.putText(frame, 'Crosstrack Error:' + str('%.2f'%error),(20,70), cv2.FONT_HERSHEY_SIMPLEX, 0.7,(255,255,255),2)
            cv2.putText(frame, 'Angle:' + str(angle[0]),(20,30), cv2.FONT_HERSHEY_SIMPLEX, 0.7,(255,255,255),2)

        lt.update(lanes)

        cv2.imshow('', frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

if __name__ == "__main__":
    args = parser.parse_args()
    main(args.path)
