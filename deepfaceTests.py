from deepface import DeepFace
from pprint import pprint
import cv2

input_folder = 'inps/'

def analyze_pic(path):
    embeddings = DeepFace.represent(img_path = path)

    print(len(embeddings))

    vec = embeddings[0]['embedding']
    face = embeddings[0]['facial_area']

    return vec, face_area


def draw_face_rect(img, face_area):
    lx = face_area["x"]
    rx = lx + face_area["w"]
    ty = face_area["y"]
    by = ty + face_area["h"]

    new_im = cv2.rectangle(img, (ty, lx), (by, rx), (255, 0, 0), 2)
    cv2.imwrite("rect.png")

v, fa = analyze_pic(f'{input_folder}pep1.JPG')
# analyze_pic(f'{input_folder}pep_daanBril.JPG')





