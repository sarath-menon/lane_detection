#!/usr/bin/env python
from __future__ import print_function
from __future__ import division

import sys
import rospy
import cv2
from std_msgs.msg import String
from sensor_msgs.msg import Image
from cv_bridge import CvBridge, CvBridgeError

import track
import detect
import argparse
import numpy as np

class Lane_Detector:

  def __init__(self):
    self.lane_image_pub = rospy.Publisher("lane_detected_image", Image, queue_size=10)
    self.bridge = CvBridge()

  def main(self):
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
        predicted = lt.predict(dt)
        lanes = ld.detect(frame)


        if predicted is not None:
          cv2.line(frame, (predicted[0][0], predicted[0][1]), (predicted[0][2], predicted[0][3]), (0, 0, 255), 7)
          cv2.line(frame, (predicted[1][2], predicted[1][3]), (predicted[1][0], predicted[1][1]), (0, 0, 255), 7)

          center_coordinates_top = ((abs(predicted[1][0]-predicted[0][2]) / 2) + predicted[0][2], predicted[1][1])
          center_coordinates_bottom = ((abs(predicted[0][0]-predicted[1][2]) / 2) + predicted[0][0], predicted[0][1])

          cv2.line(frame, center_coordinates_top, center_coordinates_bottom, (0, 2555, 0), 7)

          error = frame.shape[1]/2 - (abs(predicted[0][0]-predicted[1][2]) / 2 + predicted[0][0])
          a = [0,352]
          b = [center_coordinates_bottom[0], center_coordinates_bottom[1]]
          c = [center_coordinates_top[0],center_coordinates_top[1]]

          ba = np.subtract(a,b)
          bc = np.subtract(c,b)

          cosine_angle = np.dot(ba, bc) / (np.linalg.norm(ba) * np.linalg.norm(bc))
          angle = 90. - np.degrees(np.arccos(cosine_angle))

          cv2.putText(frame, 'Crosstrack Error:' + str('%.2f'%error),(20,70), cv2.FONT_HERSHEY_SIMPLEX, 0.7,    (255,255,255),2)
          cv2.putText(frame, 'Angle:' + str(angle[0]),(20,30), cv2.FONT_HERSHEY_SIMPLEX, 0.7,(255,255,255),2)

        lt.update(lanes)
        try:
          self.lane_image_pub.publish(self.bridge.cv2_to_imgmsg(frame, "bgr8"))
        except CvBridgeError as e:
          print(e)

if __name__ == "__main__":
    #args = parser.parse_args()
    ic = Lane_Detector()
    rospy.init_node('lane_detect_node', anonymous=True)
    ic.main()
  
    try:
      rospy.spin()
    except KeyboardInterrupt:
      print("Shutting down")
    cv2.destroyAllWindows()
    
