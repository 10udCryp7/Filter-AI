import cv2
from PIL import Image
import numpy as np
import torch
def detect_faces(image):
    """Nhận diện khuôn mặt trong ảnh (đầu vào là đối tượng Image) và trả về ảnh đã được vẽ khung."""

    # Chuyển đổi ảnh từ Pillow sang OpenCV
    image_cv = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)

    # Tải bộ phân loại khuôn mặt được huấn luyện sẵn (Haar Cascade)
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

    # Phát hiện khuôn mặt
    faces = face_cascade.detectMultiScale(image_cv, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

    # Vẽ hình chữ nhật quanh các khuôn mặt được phát hiện
    for (x, y, w, h) in faces:
        cv2.rectangle(image_cv, (x, y), (x+w, y+h), (0, 255, 0), 2)

    # Chuyển đổi ảnh trở lại từ OpenCV sang Pillow
    image_result = Image.fromarray(cv2.cvtColor(image_cv, cv2.COLOR_BGR2RGB))

    # Trả về ảnh kết quả
    return image_result, faces

def create_triangle(rect_location, landmarks_list):
    # Assuming landmarks is your list of 68 landmark points (x, y)
    landmarks = landmarks_list 

    # Convert landmarks to NumPy array
    # landmarks = np.array(landmarks, dtype=np.int32)

    # Create a Subdiv2D object for Delaunay triangulation
    rect = (0,0,1000,1000) # Adjust based on your image size
    subdiv = cv2.Subdiv2D(rect)
    # Insert landmark points into the subdiv
    for point in landmarks:
        
        subdiv.insert(point)
        
    # Get Delaunay triangles
    triangles = subdiv.getTriangleList()
    # Convert triangle points to integers
    triangles = np.array(triangles, dtype=np.int32)
    return triangles

from PIL import Image, ImageDraw

def inference(image_input, model_load, simple_transform):
    device = torch.device("cuda:0")
    image,faces = detect_faces(image_input)
    draw = ImageDraw.Draw(image_input)
    for face in faces:
        x_f, y_f, w_f, h_f = list(face)
        image_crop = image_input.crop(box=(x_f,y_f,x_f + w_f, y_f + h_f))
        image_tensor = simple_transform(image_crop).unsqueeze(0)
        input_ = image_tensor
        model_load.eval()
        with torch.inference_mode():
            o = model_load(input_)
            o = torch.squeeze(o, axis=0)
            l = (np.array(o) + 0.5) * np.array( [w_f * 224 / 256, h_f * 224 / 256] ) + [16,16] + [x_f,y_f]
            for (x_, y_) in l:
                draw.ellipse((x_-2, y_-2, x_+2, y_+2), fill='blue', outline='blue')
            triangles = create_triangle((x_f,y_f,x_f + w_f, y_f + h_f), l)
            for t in triangles:
                pt1 = (t[0], t[1])
                pt2 = (t[2], t[3])
                pt3 = (t[4], t[5])
                draw.line([pt1, pt2, pt3, pt1], fill='red', width=2)
    return image_input