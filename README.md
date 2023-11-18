<h1>Python (openCV) Lane Detection</h1>
<br>

```python
import cv2
import numpy as np
```

These lines import the necessary libraries: `cv2` for computer vision operations and `numpy` for numerical operations.

```python
def region_of_interest(img, vertices):
    mask = np.zeros_like(img)
    cv2.fillPoly(mask, vertices, 255)
    masked_img = cv2.bitwise_and(img, mask)
    return masked_img
```

The `region_of_interest` function creates a mask for the region of interest in the image defined by the `vertices` parameter. It's used to focus on the area of the image where the lane lines are expected.

```python
def draw_lines(img, lines, color=(0, 255, 0), thickness=3):
    for line in lines:
        for x1, y1, x2, y2 in line:
            cv2.line(img, (x1, y1), (x2, y2), color, thickness)
```

The `draw_lines` function draws lines on the image. It takes a list of lines (`lines`) and draws each line on the input image (`img`) using `cv2.line`.

```python
def process_image(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (5, 5), 0)
    edges = cv2.Canny(blur, 50, 150)

    height, width = edges.shape
    vertices = np.array([[(0, height), (width / 2, height / 2), (width, height)]], dtype=np.int32)
    roi = region_of_interest(edges, vertices)

    lines = cv2.HoughLinesP(roi, 1, np.pi/180, threshold=50, minLineLength=100, maxLineGap=30)

    if lines is not None:
        line_image = np.zeros_like(image)
        draw_lines(line_image, lines)
        result = cv2.addWeighted(image, 0.8, line_image, 1, 0)
    else:
        result = image

    return result
```

The `process_image` function takes an input image and performs the following steps:
- Convert the image to grayscale.
- Apply Gaussian blur to the grayscale image to reduce noise.
- Use the Canny edge detector to find edges in the image.
- Define a region of interest (ROI) to focus on a specific area of the image where the lane lines are expected.
- Use the Hough transform to detect lines in the ROI.
- Draw the detected lines on a black image (`line_image`).
- Combine the original image with the lines using `cv2.addWeighted`.

```python
cap = cv2.VideoCapture('path_to_your_video.mp4')

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    processed_frame = process_image(frame)

    cv2.imshow('Lane Detection', processed_frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
```

The main part of the code captures frames from a video file, processes each frame using the `process_image` function, and displays the result in a window. The loop continues until the 'q' key is pressed. Make sure to replace 'video.mp4' with the actual path to your video file.
