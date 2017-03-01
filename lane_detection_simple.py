# -*- coding: utf-8 -*-
"""
Created on Mon Feb 27 22:03:01 2017

@author: yangshiyu89
"""

import cv2
import numpy as np

def get_edge(frame):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray,(7,7),0)
    canny = cv2.Canny(blur, 50, 150)
    return canny

def get_roi(edge_frame):
    point1 = (round(edge_frame.shape[1]*0.18), round(edge_frame.shape[0]*0.90))
    point2 = (round(edge_frame.shape[1]*0.43), round(edge_frame.shape[0]*0.67))
    point3 = (round(edge_frame.shape[1]*0.60), round(edge_frame.shape[0]*0.67))
    point4 = (round(edge_frame.shape[1]*0.95), round(edge_frame.shape[0]*0.90))
    roi = np.array([[point1, point2, point3, point4]])
    mask = np.zeros_like(edge_frame)
    mask_value = 100
    cv2.fillPoly(mask, roi, mask_value)
    masked_img = cv2.bitwise_and(edge_frame, mask)
    return masked_img

def get_hough_lines(roi_frame):
    lines = cv2.HoughLinesP(roi_frame, 1, np.pi/180, 15, np.array([]), 
                            minLineLength=200, maxLineGap=200)
    hough_frame = np.zeros((roi_frame.shape[0], roi_frame.shape[1], 3), dtype=np.uint8)
    for line in lines:
        for x1, y1, x2, y2 in line:
            cv2.line(hough_frame, (x1, y1), (x2, y2), color=[0, 255, 0], thickness=3)
    
    return hough_frame, lines

def get_lines(frame, hough_frame, lines, threshold=0.1):
    def fit_line(lines):
        lines = np.array(lines)
        x = np.concatenate((lines[:, 0], lines[:, 2]))
        y = np.concatenate((lines[:, 1], lines[:, 3]))
        ymin, ymax = y.min(), y.max()
        model = np.polyfit(y, x, 1)
        model_poly1d = np.poly1d(model)
        xmin = int(model_poly1d(ymin))
        xmax = int(model_poly1d(ymax))
        line = [xmin, ymin, xmax, ymax]
        return line

    if len(lines) > 0:
        left_lines = []
        right_lines= []
        
        slopes = [(y2 - y1) / (x2 - x1) for line in lines for x1, y1, x2, y2 in line]
        lines = [(x1, y1, x2, y2) for line in lines for x1, y1, x2, y2 in line]
        for index in range(len(slopes)):
            if abs(slopes[index]) < threshold:
                lines.pop(lines[index])
                slopes.pop(slopes[index])
                
            else:
                if slopes[index] < 0:
                    left_lines.append(lines[index])
                else:
                    right_lines.append(lines[index])
        left_line = fit_line(left_lines)
        right_lines = fit_line(right_lines)
        cv2.line(frame, (left_line[0], left_line[1]), (left_line[2], left_line[3]), color=[255, 0, 0], thickness=3)
        cv2.line(frame, (right_lines[0], right_lines[1]), (right_lines[2], right_lines[3]), color=[0, 0, 255], thickness=3)
        
    return frame

if __name__ == "__main__":
    file_name = "lane_detection.mp4"
    cap = cv2.VideoCapture(file_name)
    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    out = cv2.VideoWriter('output.mp4',fourcc, 20.0, (640,480))
    
    while(True): 
        _, frame = cap.read()
        
        try:
            edge_frame = get_edge(frame)
            roi_frame = get_roi(edge_frame)
            hough_frame, lines = get_hough_lines(roi_frame)
            frame_result = get_lines(frame, hough_frame, lines)
            out.write(frame_result)

            cv2.imshow("frame", frame_result)
        except:
            pass
        if cv2.waitKey(10) & 0xFF == ord('q'):
            break
    cap.release()
    out.release()
    cv2.destroyAllWindows()
