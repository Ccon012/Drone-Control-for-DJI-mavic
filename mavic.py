#Credit to WeBots for the Mavic controller in C for robot movement: https://github.com/cyberbotics/webots/blob/released/projects/robots/dji/mavic/controllers/mavic2pro/mavic2pro.c 
import math
import time
import random
import copy
import cv2
import numpy as np
from matplotlib import pyplot as plt
from controller import Robot, Motor, CameraRecognitionObject, DistanceSensor, Camera, InertialUnit, GPS, Compass, Gyro, Keyboard, Lidar, Display, RangeFinder, LED
LIDAR_SENSOR_MAX_RANGE = 3 # Meters
LIDAR_ANGLE_BINS = 21 # 21 Bins to cover the angular range of the lidar, centered at 10
LIDAR_ANGLE_RANGE = 1.5708 # 90 degrees, 1.5708 radians
pose_x = 0.197
pose_y = 0.678
pose_theta = 0 
robot=Robot()
timestep = int(robot.getBasicTimeStep())

camera = robot.getDevice("camera");
Camera.enable(camera, timestep)
Camera.recognitionEnable(camera,timestep)
width = camera.getWidth()
height = camera.getHeight()
def CLAMP(value, low, high):
    if value < low: 
        return (low)
    elif (value) > (high):
        return high 
    else:
        return (value)

#camera.enable(camera, timestep);
front_left_led = robot.getDevice("front left led");
front_right_led = robot.getDevice("front right led");
imu = robot.getDevice("inertial unit");
gps = robot.getDevice("gps")
InertialUnit.enable(imu, timestep)
GPS.enable(gps, timestep)
compass = robot.getDevice("compass")
Compass.enable(compass, timestep)
gyro = robot.getDevice("gyro")
Gyro.enable(gyro, timestep);
keyboard =Keyboard()
Keyboard.enable(keyboard, timestep);
display=robot.getDevice('display')
camera_roll_motor = robot.getDevice("camera roll");
camera_pitch_motor = robot.getDevice("camera pitch");
front_left_motor = robot.getDevice("front left propeller");
front_right_motor = robot.getDevice("front right propeller");
rear_left_motor = robot.getDevice("rear left propeller");
rear_right_motor = robot.getDevice("rear right propeller");
motors = [front_left_motor, front_right_motor, rear_left_motor, rear_right_motor];
for m in range(4):
    Motor.setPosition(motors[m], float('inf'));
    Motor.setVelocity(motors[m], 1.0);
print("Start the drone...\n");
print("You can control the drone with your computer keyboard:\n");
print("- 'up': move forward.\n");
print("- 'down': move backward.\n");
print("- 'right': turn right.\n");
print("- 'left': turn left.\n");
print("- 'shift + up': increase the target altitude.\n");
print("- 'shift + down': decrease the target altitude.\n");
print("- 'shift + right': strafe right.\n");
print("- 'shift + left': strafe left.\n");

k_vertical_thrust = 68.5;  # with this thrust, the drone lifts.
k_vertical_offset = 0.6;   # Vertical offset where the robot actually targets to stabilize itself.
k_vertical_p = 3.0;        # P constant of the vertical PID.
k_roll_p = 50.0;           # P constant of the roll PID.
k_pitch_p = 30.0;          # P constant of the pitch PID.

target_altitude = 5;  # The target altitude. Can be changed by the user.
cv2.startWindowThread()
cv2.namedWindow("preview")
while (robot.step(timestep) != -1):
    if robot.getTime() > 1.0:
        break;
def doNothing(x):
    pass
# credit to Khawar Jamil for the openCV trackbar useage and object detection: https://medium.com/globant/maneuvering-color-mask-into-object-detection-fce61bf891d1 
cv2.namedWindow('Track Bars', cv2.WINDOW_NORMAL)
cv2.createTrackbar('min_blue', 'Track Bars', 0, 255, doNothing)
cv2.createTrackbar('min_green', 'Track Bars', 0, 255, doNothing)
cv2.createTrackbar('min_red', 'Track Bars', 0, 255, doNothing)

cv2.createTrackbar('max_blue', 'Track Bars', 0, 255, doNothing)
cv2.createTrackbar('max_green', 'Track Bars', 0, 255, doNothing)
cv2.createTrackbar('max_red', 'Track Bars', 0, 255, doNothing)
Motor.setPosition(camera_pitch_motor, 1.5);
obj = 'goal'
while (robot.step(timestep) != -1):
    data = camera.getImage()
    if data:
        image = np.frombuffer(data, np.uint8).reshape((camera.getHeight(), camera.getWidth(), 4))
        img = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
        if obj is 'test':
            min_blue = cv2.getTrackbarPos('min_blue', 'Track Bars')
            min_green = cv2.getTrackbarPos('min_green', 'Track Bars')
            min_red = cv2.getTrackbarPos('min_red', 'Track Bars')
            max_blue = cv2.getTrackbarPos('max_blue', 'Track Bars')
            max_green = cv2.getTrackbarPos('max_green', 'Track Bars')
            max_red = cv2.getTrackbarPos('max_red', 'Track Bars')
        if obj is 'goal':
            min_blue = 54
            min_green = 179
            min_red = 232
            max_blue = 71
            max_green = 255
            max_red = 236
        if obj is 'obstacles':
            min_blue = 116
            min_green = 186
            min_red = 226
            max_blue =144
            max_green =255
            max_red =255
        mask = cv2.inRange(hsv, (min_blue, min_green, min_red), (max_blue, max_green, max_red))
        cv2.imshow('Mask Image', mask)
        key = cv2.waitKey(timestep)
        if mask is not None:
            #credit to Opencv documentation for contour checking and position approximation: https://docs.opencv.org/3.1.0/df/d9d/tutorial_py_colorspaces.html 
            contours,hierarchy = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)
            if contours:
                for cnt in contours:
                    M = cv2.moments(cnt)
                    if M['m00'] != 0:
                        cx = int(M['m10']/M['m00'])
                        cy = int(M['m01']/M['m00'])
                        print(cx,cy)
                    area = cv2.contourArea(cnt)
                    perimeter = cv2.arcLength(cnt,True)
                    epsilon = 0.1*cv2.arcLength(cnt,True)
                    approx = cv2.approxPolyDP(cnt,epsilon,True)
                    x,y,w,h = cv2.boundingRect(cnt)
                    cv2.rectangle(image,(x,y),(x+w,y+h),(0,255,0),2)
                cv2.imshow("preview", image)
                cv2.waitKey(timestep)
    time = robot.getTime();  # in seconds.

    roll = InertialUnit.getRollPitchYaw(imu)[0] + np.pi / 2.0;
    pitch = InertialUnit.getRollPitchYaw(imu)[1];
    altitude = GPS.getValues(gps)[1];
    roll_acceleration = Gyro.getValues(gyro)[0];
    pitch_acceleration = Gyro.getValues(gyro)[1];
    num_objects = camera.getRecognitionNumberOfObjects()
    objects = camera.getRecognitionObjects()
    # Stabilize the Camera by actuating the camera motors according to the gyro feedback.
    Motor.setPosition(camera_roll_motor, -0.115 * roll_acceleration);
    Motor.setPosition(camera_pitch_motor, 1.5);

    roll_disturbance = 0.0;
    pitch_disturbance = 0.0;
    yaw_disturbance = 0.0;
    key = keyboard.getKey();
    
    while (key > 0): 
        if(key == Keyboard.UP):
            pitch_disturbance = 2.0;
            break;
        elif key==Keyboard.DOWN:
          pitch_disturbance = -2.0;
          break;
        elif key==Keyboard.RIGHT:
          yaw_disturbance = 1.3;
          break;
        elif key==Keyboard.LEFT:
          yaw_disturbance = -1.3;
          break;
        elif (key == Keyboard.SHIFT + Keyboard.RIGHT):
          roll_disturbance = -1.0;
          break;
        elif (key == Keyboard.SHIFT + Keyboard.LEFT):
          roll_disturbance = 1.0;
          break;
        elif (key == Keyboard.SHIFT + Keyboard.UP):
          target_altitude  += 0.05;
          print("target altitude: %f [m]\n", target_altitude);
          break;
        elif (key == Keyboard.SHIFT + Keyboard.DOWN):
          target_altitude -= 0.05;
          print("target altitude: %f [m]\n", target_altitude);
          break;
    key = keyboard.getKey();
      
    #Compute the roll, pitch, yaw and vertical inputs.
    roll_input = k_roll_p * CLAMP(roll, -1.0, 1.0) + roll_acceleration + roll_disturbance;
    pitch_input = k_pitch_p * CLAMP(pitch, -1.0, 1.0) - pitch_acceleration + pitch_disturbance;
    yaw_input = yaw_disturbance;
    clamped_difference_altitude = CLAMP(target_altitude - altitude, -1, 1);
    vertical_input = k_vertical_p * pow(clamped_difference_altitude, 3.0);

    # Actuate the motors taking into consideration all the computed inputs.
    front_left_motor_input = k_vertical_thrust + vertical_input - roll_input - pitch_input + yaw_input;
    front_right_motor_input = k_vertical_thrust + vertical_input + roll_input - pitch_input - yaw_input;
    rear_left_motor_input = k_vertical_thrust + vertical_input - roll_input + pitch_input - yaw_input;
    rear_right_motor_input = k_vertical_thrust + vertical_input + roll_input + pitch_input + yaw_input;
    Motor.setVelocity(front_left_motor, front_left_motor_input);
    Motor.setVelocity(front_right_motor, -front_right_motor_input);
    Motor.setVelocity(rear_left_motor, -rear_left_motor_input);
    Motor.setVelocity(rear_right_motor, rear_right_motor_input);

wb_robot_cleanup();
