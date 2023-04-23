from deepface import DeepFace
from pprint import pprint
import cv2
import numpy as np
import argparse

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

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-t','--target', help='Image path containing target face.', required=True)
    parser.add_argument('-i','--input', help='Folder path containing images to check against.', required=True)
    parser.add_argument('-o','--output', help='Name of the folder containing the output images.', default='ouput')
    parser.add_argument('-b','--blur', help='Blur mode: True to blur target face, False to blur all faces except target face. Default is True.', default=True)
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument('-c','--cosine', action='store_true', help='Cosine similarity as face recognition mode.')
    group.add_argument('-e','--euclidean', action='store_true',  help='Euclidean distance as face recognition mode.')
    args = vars(parser.parse_args())

    global target_im, inp_folder, outp_folder, blur_mode, rec_mode
    target_im = args['target']
    inp_folder = args['input']
    outp_folder = args['output']
    if args['blur']:
        blur_mode = 'target'
    else:
        blur_mode = 'others'
    if args['cosine']:
        rec_mode = 'cosine'
    else:
        rec_mode = 'euclidean'
    #mainFaceV = find_faces(f'{input_folder}ravi_pep.JPG')[0]['embedding']
    #faces = find_faces(f'{input_folder}ravi_daan.JPG')
    #for i, face in enumerate(faces):
    #    v = face["embedding"]
    #    fa = face["facial_area"]
    #    print("drawing face", i)
    #    # If face is not similar, break.
    #    if not is_similar(np.array(mainFaceV), np.array(v), mode='euclidean') :
    #        continue
    #    im = draw_face_rect(f'{input_folder}ravi_daan.JPG', fa)
    #    cv2.imwrite(f"rect{i}.png", im)




