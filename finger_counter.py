import cv2
import numpy as np
from sklearn.metrics import pairwise


class Finger_Counter:
    """Class for counting fingers on a hand placed in specified region of interest (ROI).

    # Attributes:
        background : ndarray, accumulated background of captured frame
        alpha : float, speed of updating wieght calculation
        roi_left : int, left boundary of ROI
        roi_right : int, right boundary of ROI
        roi_top : int, top boundary of ROI
        roi_bottom : int, bottom boundary of ROI
    
    # Methods:
        calc_accum_avg(frame):
            Calculates moving average of the background and updates it.

        segment(frame, threshold=25):
            Extracts the hand (if present) from captured frame and returns it.   
        
        count_fingers(thresholded, hand_segment):
            Returns number of fingers counted on captured hand segment.

        count_fingers_2(thresholded, hand_segment):
            Returns the number of found fingers (improved version not assuming hand orientation).   

        start_counting():
            Starts video capture and counts number of fingers.  
    """
    def __init__(self, left=500, right=200, top=100, bottom=350):
        """Init.

        Parameters:
            roi_left : int, left boundary of ROI
            roi_right : int, right boundary of ROI
            roi_top : int, top boundary of ROI
            roi_bottom : int, bottom boundary of ROI
        """
        self.background = None
        self.alpha = 0.5
        self.roi_left = left
        self.roi_right = right
        self.roi_top = top
        self.roi_bottom = bottom
        

    def calc_accum_avg(self, frame):
        """Calculates moving average of the background and updates it.

        Parameters:
            frame : ndarray, image captured from a camera
        """
        if self.background is None:
            self.background = frame.copy().astype(float)
            return None
        # calculate moving average of the background
        cv2.accumulateWeighted(frame, self.background, self.alpha)
        
        
    def segment(self, frame, threshold=25):
        """Extracts the hand (if present) from captured frame and returns it.

        Hand segment is obtained by calculating the difference between current frame and background average and 
        applying binary threshold. Thresholded image is also returned for inspection. 

        Parameters:
            frame : ndarray, image captured from a camera
            threshold : int, threshold value

        Returns:
            thresholded : ndarray, image after thresholding
            hand_segment : ndarray, points forming detected hand contour (None if hand not found)
        """
        diff = cv2.absdiff(self.background.astype('uint8'), frame)
        _, thresholded = cv2.threshold(diff, threshold, 255, cv2.THRESH_BINARY)
        contours, hierarchy = cv2.findContours(thresholded.copy(),
                                               cv2.RETR_EXTERNAL,
                                               cv2.CHAIN_APPROX_SIMPLE)
        # if there are no contours, hand was not detected
        if len(contours) == 0:
            return None
        # assume the largest external contour in ROI is the hand
        hand_segment = max(contours, key = cv2.contourArea)
        print(thresholded.shape, hand_segment.shape)
        return (thresholded, hand_segment)
        

    def count_fingers(self, thresholded, hand_segment):
        """Returns number of fingers counted on captured hand segment.

        Fingers are counted by creating hand's envelope (convex hull) and assuming that fingertips will be 
        farther than 80% of the distance from the centre of the convex hull to farthest point outwards.
        The method assumes that the hand is in vertical position with fingers pointing upwards. 

        Parameters:
            thresholded : ndarray, image after thresholding
            hand_segment : ndarray, points forming detected hand contour (None if hand not found)

        Returns:
            count : int, number of detected fingers
        """
        conv_hull = cv2.convexHull(hand_segment)
        # get the farthest points of convex hull in each direction
        top = tuple(conv_hull[conv_hull[:,:,1].argmin()][0])
        bottom = tuple(conv_hull[conv_hull[:,:,1].argmax()][0])
        left = tuple(conv_hull[conv_hull[:,:,0].argmin()][0])
        right = tuple(conv_hull[conv_hull[:,:,0].argmax()][0])
        # calculate the centre of rectangle defined by those points
        cX = (left[0] + right[0])//2
        cY = (top[1] + bottom[1])//2
        # find the farthest point from the centre
        distance = pairwise.euclidean_distances([(cX, cY)], Y=[left, right, top, bottom])[0]
        max_distance = distance.max()
        radius = int(0.8 * max_distance)
        circumference = (2*np.pi*radius)
        # create auxiliary circular ROI to detect contours outside of assumed 80% maximum distance
        circular_roi = np.zeros(thresholded.shape[:2], dtype='uint8')
        cv2.circle(circular_roi, (cX, cY), radius, 255, 10)
        circular_roi = cv2.bitwise_and(thresholded,thresholded,mask = circular_roi)
        contours, hierarchy = cv2.findContours(circular_roi.copy(),
                                               cv2.RETR_EXTERNAL,
                                               cv2.CHAIN_APPROX_NONE)
        count = 0
        for cnt in contours:
            (x, y, w, h) = cv2.boundingRect(cnt)
            # check if points are not located on the wrist (in the bottom of ROI)
            out_of_wrist = (cY + (cY*0.25)) > (y+h)
            if out_of_wrist:
                count +=1
        return count


    def count_fingers_2(self, thresholded, hand_segment):
        """Returns the number of found fingers (improved version not assuming hand orientation).

        Fingers are counted by finding defects in convex hull (spots between fingers on the palm of hand).
        After locating those spots, cosine theorem is used to check if angle between potential fingers is smaller
        than 90 degrees, if so the spot is couted as being between fingers.
        
        Parameters:
        thresholded : ndarray, image after thresholding
        hand_segment : ndarray, points forming detected hand contour (None if hand not found)

        Returns:
            count : int, number of detected fingers
        """
        conv_hull = cv2.convexHull(hand_segment, returnPoints=False)
        # get subset of convex hull containg defects (returns their indices)
        defects = cv2.convexityDefects(hand_segment, conv_hull)
        count = 0
        if defects is not None:
            for i in range(defects.shape[0]):
                s, e, f, d = defects[i][0]
                # get the points from hand contour by returned indices
                start = tuple(hand_segment[s][0])
                end = tuple(hand_segment[e][0])
                far = tuple(hand_segment[f][0])
                # calculate cosine theorem and check if angle is less than 90 degrees
                a = np.sqrt((end[0] - start[0]) ** 2 + (end[1] - start[1]) ** 2)
                b = np.sqrt((far[0] - start[0]) ** 2 + (far[1] - start[1]) ** 2)
                c = np.sqrt((end[0] - far[0]) ** 2 + (end[1] - far[1]) ** 2)
                angle = np.arccos((b ** 2 + c ** 2 - a ** 2) / (2 * b * c))
                if angle <= np.pi/2:
                    count += 1
            # increment by 1, as the method counts points between fingers
            if count > 0:
                count += 1
        return count
        

    def start_counting(self):
        """Starts video capture and counts number of fingers."""
        cam = cv2.VideoCapture(0)
        num_frames = 0

        while True:
            ret, frame = cam.read()
            # flip the frame so that it is not the mirror view
            frame = cv2.flip(frame, 1)
            frame_copy = frame.copy()
            # cut the region of interest
            roi = frame[self.roi_top:self.roi_bottom, self.roi_right:self.roi_left]
            # convert to greyscale and blur image to nullify irregularities
            gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
            gray = cv2.GaussianBlur(gray, (7, 7), 0)
            # for the first 60 frames calculate the average of the background
            if num_frames < 60:
                self.calc_accum_avg(gray)
                cv2.putText(frame_copy, 'Wait. getting background', (self.roi_right, self.roi_bottom+30),
                                cv2.FONT_HERSHEY_SIMPLEX, 1, (0,0,255), 1)
                num_frames += 1
            else:
                hand = self.segment(gray)
                if hand is not None:
                    # unpack thresholded image and hand segment
                    thresholded, hand_segment = hand
                    # draw contours of captured hand
                    cv2.drawContours(frame_copy, [hand_segment+(self.roi_right, self.roi_top)], -1, (255,0,0), 1)
                    fingers = self.count_fingers_2(thresholded, hand_segment)
                    cv2.putText(frame_copy, 'detected: '+str(fingers), (70,45),
                                cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255,255), 2)
                    cv2.imshow('Thresholded', thresholded)
                    cv2.waitKey(1)
            # draw ROI rectangle
            cv2.rectangle(frame_copy, (self.roi_left, self.roi_top), (self.roi_right, self.roi_bottom),
                          (0,0,255), 4)
            cv2.imshow('Finger Count', frame_copy)
            k = cv2.waitKey(1) & 0xFF
            if k ==27:
                break

        cam.release()
        cv2.destroyAllWindows()
  

# create instance of the class and start counting fingers        
counter = Finger_Counter()
counter.start_counting()        
            
