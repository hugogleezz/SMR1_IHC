import cv2
import numpy as np
import pyrealsense2 as rs
import socket
import time

def nothing(x):
    pass

# This function detects the object in the workspace and obtains its coordinates
def general_image():
    # Initialize the RealSense pipeline
    pipeline = rs.pipeline()
    config = rs.config()
    config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)
    config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)

    # Create a window for trackbars
    cv2.namedWindow("Trackbars")
    cv2.resizeWindow("Trackbars", 300, 225)
    cv2.createTrackbar("L - H", "Trackbars", 0, 179, nothing)
    cv2.createTrackbar("L - S", "Trackbars", 0, 255, nothing)
    cv2.createTrackbar("L - V", "Trackbars", 190, 255, nothing)
    cv2.createTrackbar("U - H", "Trackbars", 179, 179, nothing)
    cv2.createTrackbar("U - S", "Trackbars", 255, 255, nothing)
    cv2.createTrackbar("U - V", "Trackbars", 255, 255, nothing)

    # Start streaming
    pipeline.start(config)

    try:
        while True:
            # Wait for a new frame
            frames = pipeline.wait_for_frames()

            # Create alignment primitive with color as its target stream
            align = rs.align(rs.stream.color)
            frames = align.process(frames)

            # Get the RGB frame
            color_frame = frames.get_color_frame()
            if not color_frame:
                continue

            # Get the depth frame
            depth_frame = frames.get_depth_frame()
            if not depth_frame:
                continue

            # Convert the color frame data to a NumPy array, allowing easy manipulation of the color image.
            color_image = np.asanyarray(color_frame.get_data())

            # Create a colorizer object for depth data to convert it into a human-readable format.
            colorizer = rs.colorizer()

            # Create a hole-filling filter object to fill gaps or holes in the depth data.
            hole_filling = rs.hole_filling_filter()

            # Process the depth frame using the hole-filling filter to fill gaps in the depth data.
            filled_depth = hole_filling.process(depth_frame)

            # Colorize the filled depth data using the colorizer to produce a visually meaningful depth image.
            colorized_depth = np.asanyarray(colorizer.colorize(filled_depth).get_data())

            # Apply Gaussian blur and convert to HSV
            blurred_frame = cv2.GaussianBlur(color_image, (5, 5), 0)
            hsv = cv2.cvtColor(blurred_frame, cv2.COLOR_BGR2HSV)

            # Get trackbar values for lower and upper HSV thresholds (change if the lighting set-up or background is changed)
            l_h = cv2.getTrackbarPos("L - H", "Trackbars")
            l_s = cv2.getTrackbarPos("L - S", "Trackbars")
            l_v = cv2.getTrackbarPos("L - V", "Trackbars")
            u_h = cv2.getTrackbarPos("U - H", "Trackbars")
            u_s = cv2.getTrackbarPos("U - S", "Trackbars")
            u_v = cv2.getTrackbarPos("U - V", "Trackbars")

            lower = np.array([l_h, l_s, l_v])
            upper = np.array([u_h, u_s, u_v])
            mask = cv2.inRange(hsv, lower, upper)

            # Find contours in the mask
            contours, _ = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)

            # Initialize centroid coordinates
            centroid_x, centroid_y = 0, 0

            # Draw contours on the color image
            for contour in contours:
                area = cv2.contourArea(contour)
                
                if area > 100 and area < 2000:
                    cv2.drawContours(color_image, [contour], -1, (0, 255, 0), 3)
                    M = cv2.moments(contour)
                    if M["m00"] != 0:
                        centroid_x = int(M["m10"] / M["m00"])
                        centroid_y = int(M["m01"] / M["m00"])

            # Get the depth value at the centroid
            depth_value = depth_frame.get_distance(centroid_x, centroid_y)

            # Map the 2D centroid to 3D coordinates
            depth_intrin = depth_frame.profile.as_video_stream_profile().intrinsics
            depth_point = rs.rs2_deproject_pixel_to_point(depth_intrin, [centroid_x, centroid_y], depth_value)

            # Print the centroid coordinates
            x, y, z = depth_point
            print(f"Centroid Coordinates (X, Y, Z): ({x:.2f}, {y:.2f}, {z:.2f}) meters")

            # Apply mask to color image
            result = cv2.bitwise_and(color_image, color_image, mask=mask)

            # Merge the color and depth frames together
            images = np.hstack((color_image, colorized_depth))

            # Draw the centroid point
            cv2.circle(images, (centroid_x, centroid_y), 5, (0, 0, 255), -1)

            # Show the two frames together
            cv2.imshow("Align", images)

            # Show mask and result
            #cv2.imshow("Mask", mask)
            #cv2.imshow("Result", result)

            key = cv2.waitKey(1)
            if key == 27:
                break

    finally:
        pipeline.stop()
        cv2.destroyAllWindows()
        return x, y, z

# This function calculates the real-world position (in millimeters) of an object in a 3D plane relative to a camera's position
def general_position(x, y, z):
    # Store the input (x, y, z) as the distance_camera tuple
    distance_camera = x, y, z

    # Calculate the distance from the camera in the X and Y dimensions, converting them to millimeters by multiplying by 1000
    distance_x_mm = distance_camera[1] * 1000
    distance_y_mm = distance_camera[0] * 1000

    # Calculate the real-world X and Y positions by subtracting the distances from pre-defined constants
    real_x = 733.8 - distance_x_mm
    real_y = -2.4 - distance_y_mm

    # Store the calculated real-world X and Y positions in new variables
    pos_close_x = real_x
    pos_close_y = real_y

    # Return the calculated X and Y positions in millimeters
    return pos_close_x, pos_close_y

# This function detects the tack welds and obtains its coordinates
def precise_image():
    # Initialize the RealSense pipeline
    pipeline = rs.pipeline()
    config = rs.config()
    config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 15)
    config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 15)

    # Create a window for trackbars
    cv2.namedWindow("Trackbars")
    cv2.resizeWindow("Trackbars", 300, 225)
    cv2.createTrackbar("L - H", "Trackbars", 0, 179, nothing)
    cv2.createTrackbar("L - S", "Trackbars", 0, 255, nothing)
    cv2.createTrackbar("L - V", "Trackbars", 0, 255, nothing)
    cv2.createTrackbar("U - H", "Trackbars", 179, 179, nothing)
    cv2.createTrackbar("U - S", "Trackbars", 255, 255, nothing)
    cv2.createTrackbar("U - V", "Trackbars", 255, 255, nothing)

    # Start streaming
    pipeline.start(config)

    try:
        # Initialize a counter for detected points
        point_counter = 0
        while True:
            # Wait for a new frame
            frames = pipeline.wait_for_frames()

            # Create alignment primitive with color as its target stream
            align = rs.align(rs.stream.color)
            frames = align.process(frames)

            # Get the RGB frame
            color_frame = frames.get_color_frame()
            if not color_frame:
                continue

            # Get the depth frame
            depth_frame = frames.get_depth_frame()
            if not depth_frame:
                continue

            # Convert the color frame data to a NumPy array, allowing easy manipulation of the color image.
            color_image = np.asanyarray(color_frame.get_data())

            # Create a colorizer object for depth data to convert it into a human-readable format.
            colorizer = rs.colorizer()

            # Create a hole-filling filter object to fill gaps or holes in the depth data.
            hole_filling = rs.hole_filling_filter()

            # Process the depth frame using the hole-filling filter to fill gaps in the depth data.
            filled_depth = hole_filling.process(depth_frame)

            # Colorize the filled depth data using the colorizer to produce a visually meaningful depth image.
            colorized_depth = np.asanyarray(colorizer.colorize(filled_depth).get_data())

            # Apply Gaussian blur and convert to HSV
            blurred_frame = cv2.GaussianBlur(color_image, (5, 5), 0)
            hsv = cv2.cvtColor(blurred_frame, cv2.COLOR_BGR2HSV)

            # Get trackbar values for lower and upper HSV thresholds (change if the lighting set-up or background is changed)
            l_h = cv2.getTrackbarPos("L - H", "Trackbars")
            l_s = cv2.getTrackbarPos("L - S", "Trackbars")
            l_v = cv2.getTrackbarPos("L - V", "Trackbars")
            u_h = cv2.getTrackbarPos("U - H", "Trackbars")
            u_s = cv2.getTrackbarPos("U - S", "Trackbars")
            u_v = cv2.getTrackbarPos("U - V", "Trackbars")

            lower = np.array([l_h, l_s, l_v])
            upper = np.array([u_h, u_s, u_v])
            mask = cv2.inRange(hsv, lower, upper)

            # Find contours in the mask
            contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

            list_points = []
            parameters_set = False

            for contour in contours:
                area = cv2.contourArea(contour)

                if area < 250 and area > 30:
                    if key == ord('s'):
                        # Set the flag to indicate that parameters have been set
                        parameters_set = True

                    cv2.drawContours(color_image, [contour], -1, (0, 255, 0), 3)
                    M = cv2.moments(contour)
                    if M["m00"] != 0:
                        cx = int(M['m10'] / M['m00'])
                        cy = int(M['m01'] / M['m00'])

                    # Get the depth value at the centroid
                    depth_value = depth_frame.get_distance(cx, cy)

                    # Map the 2D centroid to 3D coordinates
                    depth_intrin = depth_frame.profile.as_video_stream_profile().intrinsics
                    depth_point = rs.rs2_deproject_pixel_to_point(depth_intrin, [cx, cy], depth_value)

                    # Store the depth points in a list
                    list_points.append(depth_point)

                    if parameters_set == True:
                        # Increment the point counter
                        point_counter += 1

                    # Print the centroid coordinates
                    x, y, z = depth_point
                    print(f"Centroid Coordinates (X, Y, Z): ({x:.2f}, {y:.2f}, {z:.2f}) meters")

                    # Draw the centroid point on the color image in red
                    cv2.circle(color_image, (cx, cy), 5, (0, 0, 255), -1)

            # Apply mask to color image
            result = cv2.bitwise_and(color_image, color_image, mask=mask)

            # Merge the color and depth frames together
            images = np.hstack((color_image, colorized_depth))

            # Show the two frames together
            cv2.imshow("Align", images)

            # Show mask and result
            # cv2.imshow("Mask", mask)
            # cv2.imshow("Result", result)

            key = cv2.waitKey(1)
            if key == 27:
                break

    finally:
        pipeline.stop()
        cv2.destroyAllWindows()
        return list_points, point_counter

# This function calculates the precise real-world positions (in millimeters) of both a starting and an ending point in a 3D space relative to a camera's position
def precise_position(list_coordinates, pos_close_x, pos_close_y):
    # Extract individual values for the starting point (x1, y1, z1) from the list of coordinates
    x1 = list_coordinates[0][1] * 1000
    y1 = list_coordinates[0][0] * 1000
    z1 = list_coordinates[0][2] * 1000

    # Extract individual values for the ending point (x2, y2, z2) from the list of coordinates
    x2 = list_coordinates[1][1] * 1000
    y2 = list_coordinates[1][0] * 1000
    z2 = list_coordinates[1][2] * 1000

    # Calculate the real-world coordinates of the starting point by subtracting the distances (in millimeters) from the provided coordinates
    start_point_x = pos_close_x - x1
    start_point_y = pos_close_y - (y1 - 10)
    start_point_z = 350 - z1

    # Calculate the real-world coordinates of the ending point by subtracting the distances (in millimeters) from the provided coordinates
    end_point_x = pos_close_x - x2
    end_point_y = pos_close_y - (y2 - 10)
    end_point_z = 350 - z2

    # Return the calculated real-world coordinates for the starting and ending points
    return start_point_x, start_point_y, start_point_z, end_point_x, end_point_y, end_point_z

# This function calculates the sleep time required for a welding operation based on the starting and ending positions in a 3D space.
def sleep_time(start_point_x, start_point_y, start_point_z, end_point_x, end_point_y, end_point_z):
    # Calculate the Euclidean distance between the starting and ending positions in 3D space
    distance = np.sqrt((end_point_x - start_point_x)**2 + (end_point_y - start_point_y)**2 + (end_point_z - start_point_z)**2)

    # Calculate the sleep time required for the welding operation by dividing the distance by 7
    # This division assumes a constant velocity of 7 units per unit of distance (e.g., millimeters per second)
    sleeptime = distance / 7

    # Return the calculated sleep time, which represents the time required to move from the starting position to the ending position at a constant velocity of 7 units per unit of distance.
    return sleeptime

def main():
    # Create a TCP/IP socket
    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)

    # Connect the socket to the port where the robot is listening
    robot_address = ('192.168.137.100', 9225)
    sock.connect(robot_address)

    try:
        # Welding side (change based on the orientation of the product)
        case = False

        # Set camera as main end-of-arm tool
        sock.sendall(b"set_tcp('camera')")
        time.sleep(2)

        sock.sendall(b"set_digital_output(1, OFF)")
        time.sleep(2)

        # Move to general view
        sock.sendall(b"movej(posj(-5.0, 20.1, 52.0, -189.1, -142.2, 30.6), v=100, a=60)")

        # Get coordinates of the object from the general view 
        x, y, z = general_image()

        # Convert the camera coordinates to robot coordinates
        pos_close_x, pos_close_y = general_position(x, y, z)

        if case == False:
            # Send data (Precise view)
            sock.sendall(f"movel(posx({pos_close_x}, {pos_close_y} + 100, 350, 4.89, -144.65, -134.07), v=100, a=80)".encode())

            # Get coordinates of the object from the precise view
            list_coordinates, point_counter = precise_image()

            # Store the points of the fist recognised side
            list_coordinates_first_side = list_coordinates

            # Move robot to see the remaining side of the product
            sock.sendall(f"movel(posx({pos_close_x}, {pos_close_y} - 50, 350, 4.89, -144.65, -134.07), v=60, a=40)".encode())

            # Get coordinates of the object from the precise view
            list_coordinates, point_counter = precise_image()

            # Set welding torch as main end-of-arm tool
            sock.sendall(b"set_tcp('welding torch')")

            # Convert the camera coordinates to robot coordinates
            start_point_x, start_point_y, start_point_z, end_point_x, end_point_y, end_point_z = precise_position(list_coordinates, pos_close_x, pos_close_y)
            sleeptime = sleep_time(start_point_x, start_point_y, start_point_z, end_point_x, end_point_y, end_point_z)

            # Move robot close to initial welding point
            sock.sendall(f"movel(posx({start_point_x}, {start_point_y} - 90, {start_point_z} + 10, 120, 125, 80), v=80, a=50)".encode())
            
            # Wait for robot to end the movement
            time.sleep(5)

            # Move robot to initial welding point
            sock.sendall(f"movel(posx({start_point_x}, {start_point_y} - 40, {start_point_z} - 5, 120, 125, 80), v=35, a=20)".encode())
            
            # Wait for robot to end the movement
            time.sleep(5)

            # Switch on the welding torch
            sock.sendall(b"set_digital_output(1, ON)")
            print("relay on")
            time.sleep(1)

            # Move robot to ending welding point
            sock.sendall(f"movel(posx({end_point_x} - 5, {end_point_y} - 40, {end_point_z} - 5, 120, 125, 80), v=7, a=15)".encode())
            
            # Wait for robot to end the movement
            time.sleep(sleeptime)

            # Switch off the welding torch
            sock.sendall(b"set_digital_output(1, OFF)")
            print("relay off")
            time.sleep(1)

            # Move robot apart not to crash with the product
            sock.sendall(f"movel(posx({end_point_x} - 2, {end_point_y} - 100, {end_point_z} + 80, 120, 125, 80), v=100, a=100)".encode())
            
            # Wait for robot to end the movement
            time.sleep(3)

            # Set camera as main end-of-arm tool
            sock.sendall(b"set_tcp('camera')")
            time.sleep(1)

            # Move robot to precise position
            sock.sendall(f"movel(posx({pos_close_x}, {pos_close_y}, 350, 4.89, -144.65, -134.07), v=100, a=100)".encode())
            
            # Wait for robot to end the movement
            time.sleep(5)

            # Set welding torch as main end-of-arm tool
            sock.sendall(b"set_tcp('welding torch')")
            time.sleep(2)

            # Convert the camera coordinates to robot coordinates
            start_point_x, start_point_y, start_point_z, end_point_x, end_point_y, end_point_z = precise_position(list_coordinates_first_side, pos_close_x, pos_close_y)
            sleeptime = sleep_time(start_point_x, start_point_y, start_point_z, end_point_x, end_point_y, end_point_z)

            # Move robot close to initial welding point
            sock.sendall(f"movel(posx({start_point_x}, {start_point_y} + 140, {start_point_z} + 10, 70.0, -125.00, 45.00), v=70, a=20)".encode())
            
            # Wait for robot to end the movement
            time.sleep(8)

            # Move robot to initial welding point
            sock.sendall(f"movel(posx({start_point_x}, {start_point_y} + 100, {start_point_z} + 5, 70.0, -125.00, 45.00), v=35, a=20)".encode())
            
            # Wait for robot to end the movement
            time.sleep(5)

            # Switch on the welding torch
            sock.sendall(b"set_digital_output(1, ON)")
            print("relay on")
            time.sleep(1)

            # Move robot to ending welding point
            sock.sendall(f"movel(posx({end_point_x} - 5, {end_point_y} + 100, {end_point_z} + 5, 70.0, -125.00, 45.00), v=7, a=15)".encode())
            
            # Wait for robot to end the movement
            time.sleep(sleeptime)

            # Switch off the welding torch
            sock.sendall(b"set_digital_output(1, OFF)")
            print("relay off")
            time.sleep(1)

            # Move robot apart not to crash with the product
            sock.sendall(f"movel(posx({end_point_x} + 5, {end_point_y} + 200, {end_point_z} + 100, 70.0, -125.00, 45.00), v=100, a=100)".encode())
            
            # Wait for robot to end the movement
            time.sleep(2)

        else:
            # Send data (Precise view)
            sock.sendall(f"movel(posx({pos_close_x} - 75, {pos_close_y} + 20, 350, 4.89, -144.65, -134.07), v=100, a=80)".encode())

            # Get coordinates of the object from the precise view
            list_coordinates, point_counter = precise_image()

            # Store the points of the fist recognised side
            list_coordinates_first_side = list_coordinates

            # Move robot to see the remaining side of the product
            sock.sendall(f"movel(posx({pos_close_x} + 55, {pos_close_y} + 20, 350, 4.89, -144.65, -134.07), v=60, a=40)".encode())

            # Get coordinates of the object from the precise view
            list_coordinates, point_counter = precise_image()

            # Set welding torch as main end-of-arm tool
            sock.sendall(b"set_tcp('welding torch')")

            # Convert the camera coordinates to robot coordinates
            start_point_x, start_point_y, start_point_z, end_point_x, end_point_y, end_point_z = precise_position(list_coordinates, pos_close_x, pos_close_y)
            sleeptime = sleep_time(start_point_x, start_point_y, start_point_z, end_point_x, end_point_y, end_point_z)

            # Move robot to precise position
            sock.sendall(f"movel(posx({start_point_x} + 150, {start_point_y}, {start_point_z} + 150, 22, -130, -100), v=100, a=80)".encode())
            
            # Wait for robot to end the movement
            time.sleep(5)

            # Move robot close to initial welding point
            sock.sendall(f"movel(posx({start_point_x} + 80, {start_point_y}, {start_point_z} + 13, 22, -130, -100), v=80, a=50)".encode())
            
            # Wait for robot to end the movement
            time.sleep(5)

            # Move robot to initial welding point
            sock.sendall(f"movel(posx({start_point_x} + 52, {start_point_y} + 30, {start_point_z} - 3, 22, -130, -100), v=35, a=20)".encode())
            
            # Wait for robot to end the movement
            time.sleep(5)

            # Switch on the welding torch
            sock.sendall(b"set_digital_output(1, ON)")
            print("relay on")
            time.sleep(1)

            # Move robot to ending welding point
            sock.sendall(f"movel(posx({end_point_x} + 52, {end_point_y} + 30, {end_point_z} - 3, 22, -130, -100), v=7, a=15)".encode())
            
            # Wait for robot to end the movement
            time.sleep(sleeptime)

            # Switch off the welding torch
            sock.sendall(b"set_digital_output(1, OFF)")
            print("relay off")
            time.sleep(1)

            # Move robot apart not to crash with the product
            sock.sendall(f"movel(posx({end_point_x} + 80, {end_point_y}, {end_point_z} + 150, 22, -130, -100), v=100, a=80)".encode())
            
            # Wait for robot to end the movement
            time.sleep(3)

            # Set camera as main end-of-arm tool
            sock.sendall(b"set_tcp('camera')")
            time.sleep(1)

            # Move robot to precise position
            sock.sendall(f"movel(posx({pos_close_x}, {pos_close_y}, 350, 4.89, -144.65, -134.07), v=100, a=80)".encode())
            
            # Wait for robot to end the movement
            time.sleep(4)

            # Set welding torch as main end-of-arm tool
            sock.sendall(b"set_tcp('welding torch')")
            time.sleep(3)

            # Convert the camera coordinates to robot coordinates
            start_point_x, start_point_y, start_point_z, end_point_x, end_point_y, end_point_z = precise_position(list_coordinates_first_side, pos_close_x, pos_close_y)
            sleeptime = sleep_time(start_point_x, start_point_y, start_point_z, end_point_x, end_point_y, end_point_z)

            # Move robot close to initial welding point
            sock.sendall(f"movel(posx({start_point_x} - 140, {start_point_y} + 8, {start_point_z} + 20, 168.54, -121.87, -156.66), v=80, a=50)".encode())
            
            # Wait for robot to end the movement
            time.sleep(5)

            # Move robot to initial welding point
            sock.sendall(f"movel(posx({start_point_x} - 84, {start_point_y} + 25, {start_point_z} - 11, 168.54, -121.87, -156.66), v=35, a=20)".encode())
            
            # Wait for robot to end the movement
            time.sleep(5)

            # Switch on the welding torch
            sock.sendall(b"set_digital_output(1, ON)")
            print("relay on")
            time.sleep(1)

            # Move robot to ending welding point
            sock.sendall(f"movel(posx({end_point_x} - 84, {end_point_y} + 27, {end_point_z} - 11, 168.54, -121.87, -156.66), v=7, a=15)".encode())
            
            # Wait for robot to end the movement
            time.sleep(sleeptime)

            # Switch off the welding torch
            sock.sendall(b"set_digital_output(1, OFF)")
            print("relay off")
            time.sleep(1)

            sock.sendall(b"set_digital_output(1, OFF)")
            time.sleep(1)

            # Move robot apart not to crash with the product
            sock.sendall(f"movel(posx({end_point_x} - 100, {end_point_y}, {end_point_z} + 75, 168.54, -121.87, -156.66), v=100, a=100)".encode())
    
            # Wait for robot to end the movement
            time.sleep(2)

        # Move to home position
        sock.sendall(b"move_home(DR_HOME_TARGET_USER)")
        time.sleep(5)

    finally:
        # Close the socket
        sock.close()

if __name__ == "__main__":
    main()