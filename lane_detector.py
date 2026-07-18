import cv2
import numpy as np
import pigpio
import time
import sys
import signal
import atexit

SERVO_PIN = 18
ENA = 25
IN1 = 24
IN2 = 23

PULSE_LEFT = 600     
PULSE_RIGHT = 2300    
PULSE_CENTER = 1500 

BASE_SPEED = 70
TURN_SPEED = 70
MIN_CONTOUR_AREA = 500 

class LaneDetector:
    def __init__(self):
        self.running = True
        self.current_pulse = PULSE_CENTER
        
        self.pi = pigpio.pi()
        if not self.pi.connected:
            raise Exception("pigpiod service is not running!")

        self.pi.set_mode(SERVO_PIN, pigpio.OUTPUT)
        self.pi.set_mode(ENA, pigpio.OUTPUT)
        self.pi.set_mode(IN1, pigpio.OUTPUT)
        self.pi.set_mode(IN2, pigpio.OUTPUT)

        self.center_servo()
        self.stop_motors()

        signal.signal(signal.SIGINT, self.signal_handler)
        atexit.register(self.cleanup)

    def signal_handler(self, sig, frame):
        self.cleanup()
        sys.exit(0)

    def cleanup(self):
        if not self.running: 
            return
        self.running = False
        print("Cleaning up...")
        self.stop_motors()
        try:
            self.pi.set_servo_pulsewidth(SERVO_PIN, PULSE_CENTER)
            time.sleep(0.5)
            self.pi.set_servo_pulsewidth(SERVO_PIN, 0)
            self.pi.stop()
        except: 
            pass

    def center_servo(self):
        self.pi.set_servo_pulsewidth(SERVO_PIN, PULSE_CENTER)
        time.sleep(0.3)
        self.pi.set_servo_pulsewidth(SERVO_PIN, 0)

    def steer(self, direction):
        target_pulse = PULSE_CENTER
        if direction == 'LEFT': 
            target_pulse = PULSE_LEFT
        elif direction == 'RIGHT': 
            target_pulse = PULSE_RIGHT
        
        if target_pulse != self.current_pulse:
            self.pi.set_servo_pulsewidth(SERVO_PIN, target_pulse)
            self.current_pulse = target_pulse
            if target_pulse == PULSE_CENTER:
                time.sleep(0.3)
                self.pi.set_servo_pulsewidth(SERVO_PIN, 0)

    def set_motor(self, speed):
        if not self.running: 
            return
        speed = max(0, min(100, speed))
        if speed > 0:
            self.pi.write(IN1, 1)
            self.pi.write(IN2, 0)
            self.pi.set_PWM_dutycycle(ENA, int(speed * 2.55))
        else:
            self.stop_motors()

    def stop_motors(self):
        self.pi.write(IN1, 0)
        self.pi.write(IN2, 0)
        self.pi.set_PWM_dutycycle(ENA, 0)

    def get_max_cohesion(self, mask):
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not contours:
            return 0, None
        largest_contour = max(contours, key=cv2.contourArea)
        area = cv2.contourArea(largest_contour)
        if area < MIN_CONTOUR_AREA:
            return 0, None
        return area, largest_contour

    def filter_noise(self, mask, min_area, keep_only_largest=False):
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        clean_mask = np.zeros_like(mask)
        
        if contours:
            if keep_only_largest:
                largest_contour = max(contours, key=cv2.contourArea)
                if cv2.contourArea(largest_contour) > min_area:
                    cv2.drawContours(clean_mask, [largest_contour], -1, 255, -1)
            else:
                for cnt in contours:
                    if cv2.contourArea(cnt) > min_area:
                        cv2.drawContours(clean_mask, [cnt], -1, 255, -1)
                    
        return clean_mask

    def detect_lanes(self, frame):
        if frame is None: 
            return frame, 0
        
        height, width = frame.shape[:2]
        center_x = width // 2
        final_visual = frame.copy()
        
        command = 'CENTER'
        speed = BASE_SPEED
        mode = "SEARCHING"

        try:
            # crop roi
            roi_height = int(height * 0.85) 
            roi_y_start = height - roi_height
            roi = frame[roi_y_start:height, :]
            
            hls = cv2.cvtColor(roi, cv2.COLOR_BGR2HLS)
            
            # clahe
            h, l, s = cv2.split(hls)
            clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
            l = clahe.apply(l)
            hls = cv2.merge((h, l, s))
            
            # color thresholds
            lower_white = np.array([0, 120, 0])
            upper_white = np.array([180, 255, 255])
            mask_white = cv2.inRange(hls, lower_white, upper_white)

            lower_black = np.array([0, 0, 0])
            upper_black = np.array([180, 100, 60])
            mask_black = cv2.inRange(hls, lower_black, upper_black)
            
            # cleanup
            kernel = np.ones((5,5), np.uint8)
            mask_black = cv2.morphologyEx(mask_black, cv2.MORPH_CLOSE, kernel)
            mask_white = cv2.morphologyEx(mask_white, cv2.MORPH_CLOSE, kernel)

            # block background above white lines
            h_roi, w_roi = mask_black.shape
            row_indices = np.arange(h_roi).reshape(h_roi, 1)
            has_white = np.any(mask_white > 0, axis=0)
            top_white_y = np.argmax(mask_white > 0, axis=0)
            
            cutoff_mask = (row_indices < top_white_y) & has_white
            mask_black[cutoff_mask] = 0

            # area filtering to kill ceramic noise and glare
            mask_black = self.filter_noise(mask_black, 5000, keep_only_largest=True)
            mask_white = self.filter_noise(mask_white, 2000, keep_only_largest=False)

            # split screen
            left_green_mask = mask_black[:, :center_x]
            right_green_mask = mask_black[:, center_x:]
            
            left_red_mask = mask_white[:, :center_x]
            right_red_mask = mask_white[:, center_x:]

            l_green_score, l_green_cnt = self.get_max_cohesion(left_green_mask)
            r_green_score, r_green_cnt = self.get_max_cohesion(right_green_mask)
            l_red_score, l_red_cnt = self.get_max_cohesion(left_red_mask)
            r_red_score, r_red_cnt = self.get_max_cohesion(right_red_mask)

            cv2.putText(final_visual, f"L-Black: {int(l_green_score)}", (10, height-80), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
            cv2.putText(final_visual, f"R-Black: {int(r_green_score)}", (width-200, height-80), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
            cv2.putText(final_visual, f"L-White: {int(l_red_score)}", (10, height-50), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
            cv2.putText(final_visual, f"R-White: {int(r_red_score)}", (width-200, height-50), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)

            RED_DANGER = 1000
            total_black = l_green_score + r_green_score

            if total_black < 5000:
                command = 'CENTER'
                speed = 0
                mode = "STOP: NO ROAD"
            else:
                left_ratio = l_green_score / total_black
                right_ratio = r_green_score / total_black
                
                cv2.putText(final_visual, f"L%: {int(left_ratio*100)}  R%: {int(right_ratio*100)}", (center_x-70, height-20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)

                if l_red_score > 15000 or r_red_score > 15000:
                    if l_red_score > r_red_score:
                        command = 'RIGHT'
                        speed = TURN_SPEED
                        mode = "EMERGENCY: AVOID L"
                    else:
                        command = 'LEFT'
                        speed = TURN_SPEED
                        mode = "EMERGENCY: AVOID R"
                        
                elif l_red_score > RED_DANGER or r_red_score > RED_DANGER:
                    if l_red_score > r_red_score + 500:
                        command = 'RIGHT'
                        speed = TURN_SPEED
                        mode = "AVOID WALL (L)"
                    elif r_red_score > l_red_score + 500:
                        command = 'LEFT'
                        speed = TURN_SPEED
                        mode = "AVOID WALL (R)"
                    else:
                        command = 'CENTER'
                        speed = BASE_SPEED
                        mode = "FORWARD (NARROW)"
                        
                elif left_ratio > 0.65:
                    command = 'LEFT'
                    speed = TURN_SPEED
                    mode = "CURVE LEFT"
                    
                elif right_ratio > 0.65:
                    command = 'RIGHT'
                    speed = TURN_SPEED
                    mode = "CURVE RIGHT"
                    
                else:
                    if l_green_score > r_green_score + 2000:
                        command = 'LEFT'
                        speed = TURN_SPEED
                        mode = "ADJUST LEFT"
                    elif r_green_score > l_green_score + 2000:
                        command = 'RIGHT'
                        speed = TURN_SPEED
                        mode = "ADJUST RIGHT"
                    else:
                        command = 'CENTER'
                        speed = BASE_SPEED
                        mode = "FORWARD"

            self.steer(command)
            self.set_motor(speed)

            offset_x = center_x
            offset_y = roi_y_start
            
            if l_green_cnt is not None:
                l_green_cnt[:, :, 1] += offset_y
                cv2.drawContours(final_visual, [l_green_cnt], -1, (0, 255, 0), 3) 
            if r_green_cnt is not None:
                r_green_cnt[:, :, 0] += offset_x 
                r_green_cnt[:, :, 1] += offset_y
                cv2.drawContours(final_visual, [r_green_cnt], -1, (0, 255, 0), 3)

            if l_red_cnt is not None:
                l_red_cnt[:, :, 1] += offset_y
                cv2.drawContours(final_visual, [l_red_cnt], -1, (0, 0, 255), 3)
            if r_red_cnt is not None:
                r_red_cnt[:, :, 0] += offset_x
                r_red_cnt[:, :, 1] += offset_y
                cv2.drawContours(final_visual, [r_red_cnt], -1, (0, 0, 255), 3)

            cv2.line(final_visual, (0, roi_y_start), (width, roi_y_start), (255, 255, 0), 2)
            cv2.line(final_visual, (center_x, roi_y_start), (center_x, height), (255, 255, 0), 2)
            
            cv2.putText(final_visual, f"CMD: {command} | SPD: {speed}", (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)
            cv2.putText(final_visual, f"Mode: {mode}", (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 255), 2)
            
            return final_visual, 0

        except Exception as e:
            print(f"Error in detect_lanes: {e}")
            self.stop_motors()
            return frame, 0