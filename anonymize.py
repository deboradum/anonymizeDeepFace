from deepface import DeepFace
from pprint import pprint
import cv2
import numpy as np
import argparse
from os import listdir
import os.path

ACCEPTED_FORMATS = ['png', 'PNG', 'jpg', 'jpeg', 'JPG', 'JPEG', 'webp']


# Finds data of all faces in the image.
def find_faces(path):
    embeddings = DeepFace.represent(img_path = path)

    return embeddings


# Draws a rectangle around the face. Used for testing.
def draw_face_rect(img_path, face_area, img=None):
    if img is None:
        img = cv2.imread(img_path)
    lx = face_area["x"]
    rx = lx + face_area["w"]
    ty = face_area["y"]
    by = ty + face_area["h"]

    new_im = cv2.rectangle(img, (lx, ty), (rx, by), (255, 0, 0), 2)
    return new_im


def draw_number(img, face_area, number):
    lx = face_area["x"]
    rx = lx + face_area["w"]
    ty = face_area["y"]     
    by = ty + face_area["h"]

    return cv2.putText(img, str(number), (lx, ty), cv2.FONT_HERSHEY_SIMPLEX, 1, (255,255,255), 3)

# Checks if a face is similar (enough) to the main face.
# Accepted modes: cosine or euclidean.
def is_similar(main_face, check_face):
    if rec_mode == 'cosine':
        cs = np.dot(main_face, check_face) / (np.linalg.norm(main_face) * np.linalg.norm(check_face))
        print(cs)
        if cs >= 0.6:
            return True
        else:
            return False
    elif rec_mode == 'euclidean':
        ed = np.linalg.norm(main_face - check_face)
        print(ed)
        if ed < 0.5:
            return True
        else:
            return False


def accepted_file(filename):
    try:
        suffix = filename.split('.')[-1]
    except Exception as e:
        return False
    if suffix in ACCEPTED_FORMATS:
        return True
    else:
        return False
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

    if not os.path.exists(outp_folder):
        os.makedirs(outp_folder)

    # Get target face vector.
    targetFace = find_faces(target_im)
    found = False
    for i, face in enumerate(targetFace):
        targetFaceV = face['embedding']
        fa = face['facial_area']
        if fa['w'] > 500 and fa['h'] > 500:
            found = True
            break
    if not found:
        print("No valid face detected in target image. Exiting.")
        exit()

    for f in listdir(inp_folder):
        path = os.path.join(inp_folder, f)
        if not os.path.isfile(path) or not accepted_file(f):
            continue
        print('scanning', f)
        try:
            faces = find_faces(path)
        except Exception as e:
            print("No faces found")
            cv2.imwrite(os.path.join(outp_folder, f), cv2.imread(path))
            continue
        im = None
        for i, face in enumerate(faces):
            v = face["embedding"]
            fa = face["facial_area"]
            # If face is not similar, break.
            if not is_similar(np.array(targetFaceV), np.array(v)) :
                continue
            print('Found face')
            im = draw_face_rect(path, fa, img=im)

        if im is not None:
            cv2.imwrite(os.path.join(outp_folder, f), im)
        else:
            cv2.imwrite(os.path.join(outp_folder, f), cv2.imread(path))

