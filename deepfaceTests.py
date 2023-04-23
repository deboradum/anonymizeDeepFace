from deepface import DeepFace
from pprint import pprint
import cv2
import numpy as np
input_folder = 'inps/'


# Finds data of all faces in the image.
def find_faces(path):
    embeddings = DeepFace.represent(img_path = path)

    #vec = embeddings[0]['embedding']
    #face = embeddings[0]['facial_area']

    #return vec, face
    return embeddings


# Draws a rectangle around the face. Used for testing.
def draw_face_rect(img_path, face_area):
    img = cv2.imread(img_path)
    lx = face_area["x"]
    rx = lx + face_area["w"]
    ty = face_area["y"]
    by = ty + face_area["h"]

    new_im = cv2.rectangle(img, (lx, ty), (rx, by), (255, 0, 0), 2)
    return new_im


# Checks if a face is similar (enough) to the main face.
# Accepted modes: cosine or euclidean.
def is_similar(main_face, check_face, mode='cosine'):
    if mode == 'cosine':
        cs = np.dot(main_face, check_face) / (np.linalg.norm(main_face) * np.linalg.norm(check_face))
        print(cs)
        if cs >= 0.6:
            return True
        else:
            return False
    elif mode == 'euclidean':
        ed = np.linalg.norm(main_face - check_face)
        print(ed)
        if ed < 0.5:
            return True
        else:
            return False
    return NotImplementedError

# --------------------------------------------------------
mainFaceV = find_faces(f'{input_folder}ravi_pep.JPG')[0]['embedding']
faces = find_faces(f'{input_folder}ravi_daan.JPG')
for i, face in enumerate(faces):
    v = face["embedding"]
    fa = face["facial_area"]
    print("drawing face", i)
    # If face is not similar, break.
    if not is_similar(np.array(mainFaceV), np.array(v), mode='cosine') :
        continue
    im = draw_face_rect(f'{input_folder}ravi_daan.JPG', fa)
    cv2.imwrite(f"rect{i}.png", im)




