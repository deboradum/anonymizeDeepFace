from deepface import DeepFace
from pprint import pprint
import cv2

input_folder = 'inps/'

def analyze_pic(path):
    embeddings = DeepFace.represent(img_path = path)

    vec = embeddings[0]['embedding']
    face = embeddings[0]['facial_area']

    return vec, face


def draw_face_rect(img_path, face_area):
    img = cv2.imread(img_path)
    lx = face_area["x"]
    rx = lx + face_area["w"]
    ty = face_area["y"]
    by = ty + face_area["h"]

    new_im = cv2.rectangle(img, (lx, ty), (rx, by), (255, 0, 0), 2)
    cv2.imwrite("rect.png", new_im)


print("Analyzing face...")
v, fa = analyze_pic(f'{input_folder}pep1.JPG')
print("drawing rectangle...")

draw_face_rect(f'{input_folder}pep1.JPG', fa)
# analyze_pic(f'{input_folder}pep_daanBril.JPG')





