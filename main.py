import cv2
import numpy as np

def region_of_interest(img, vertices):
    mask = np.zeros_like(img)
    cv2.fillPoly(mask, vertices, 255)
    masked_img = cv2.bitwise_and(img, mask)
    return masked_img

def draw_lines(img, lines, color=(0, 255, 0), thickness=3):
    for line in lines:
        for x1, y1, x2, y2 in line:
            cv2.line(img, (x1, y1), (x2, y2), color, thickness)

def process_image(image):
    # Convert the image to grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Apply Gaussian blur to the image
    blur = cv2.GaussianBlur(gray, (5, 5), 0)

    # Apply Canny edge detection
    edges = cv2.Canny(blur, 50, 150)

    # Define region of interest
    height, width = edges.shape
    vertices = np.array([[(0, height), (width / 2, height / 2), (width, height)]], dtype=np.int32)
    roi = region_of_interest(edges, vertices)

    # Apply Hough transform to find lines
    lines = cv2.HoughLinesP(roi, 1, np.pi/180, threshold=50, minLineLength=100, maxLineGap=30)


    # Check if lines were detected
    if lines is not None:
        # Draw the detected lines on the original image
        line_image = np.zeros_like(image)
        draw_lines(line_image, lines)

        # Combine the original image with the lines
        result = cv2.addWeighted(image, 0.8, line_image, 1, 0)
    else:
        # If no lines were detected, return the original image
        result = image

    return result

# Open a video capture object
cap = cv2.VideoCapture('video.mp4')

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    processed_frame = process_image(frame)

    # Display the resulting frame
    cv2.imshow('Lane Detection', processed_frame)

    # Break the loop if 'q' key is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the video capture object and close all windows
cap.release()
cv2.destroyAllWindows()
