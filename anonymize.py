from deepface import DeepFace
from pprint import pprint
import cv2
import numpy as np
import argparse
from os import listdir
import os.path

ACCEPTED_FORMATS = ['png', 'PNG', 'jpg', 'jpeg', 'JPG', 'JPEG', 'webp']


# https://stackoverflow.com/questions/55324449/how-to-specify-a-minimum-or-maximum-float-value-with-argparse
def float_range(mini,maxi):
    """Return function handle of an argument type function for
       ArgumentParser checking a float range: mini <= arg <= maxi
         mini - minimum acceptable argument
         maxi - maximum acceptable argument"""

    # Define the function with default arguments
    def float_range_checker(arg):
        """New Type function for argparse - a float within predefined range."""

        try:
            f = float(arg)
        except ValueError:
            raise argparse.ArgumentTypeError("must be a floating point number")
        if f < mini or f > maxi:
            raise argparse.ArgumentTypeError("must be in range [" + str(mini) + " .. " + str(maxi)+"]")
        return f

    # Return function handle to checking function
    return float_range_checker


# Checks if file is legal (supported image type).
def accepted_file(filename):
    try:
        suffix = filename.split('.')[-1]
    except Exception as e:
        return False
    if suffix in ACCEPTED_FORMATS:
        return True
    else:
        return False


class Recognizer():
    def __init__(self, target_path, inputs, output, blur_mode, rec_mode, threshold):
        self.targetFaceV = self.find_target(target_path)
        self.input_folder = inputs
        self.output_folder = output
        self.blur_mode = blur_mode
        self.rec_mode = rec_mode
        self.threshold = threshold


    # Finds data of all faces in the image.
    def find_faces(self, path):
        embeddings = DeepFace.represent(img_path = path)
     
        return embeddings


    def find_target(self, path):
        faces = self.find_faces(path)
        largest_size = 0
        for face in faces:
            v = face['embedding']
            fa = face['facial_area']
            if fa['w'] > largest_size:
                size = fa['w']
                best_fit = v
                                       
        return np.array(best_fit)

   
    # Draws a rectangle around the face. Used for testing.                
    def draw_face_rect(self, img_path, face_area, img=None):
        if img is None:
            img = cv2.imread(img_path)
        lx = face_area["x"]
        rx = lx + face_area["w"]
        ty = face_area["y"]
        by = ty + face_area["h"]
    
        new_im = cv2.rectangle(img, (lx, ty), (rx, by), (255, 0, 0), 2)

        return new_im


    # Checks if a face is similar (enough) to the main face.
    # Accepted modes: cosine or euclidean.
    def is_similar(self, check_face):
        if self.rec_mode == 'cosine':
            cs = np.dot(self.targetFaceV, check_face) / (np.linalg.norm(self.targetFaceV) * np.linalg.norm(check_face))
            print(cs)
            if cs >= self.threshold:
                return True
            else:
                return False
        elif self.rec_mode == 'euclidean':
            ed = np.linalg.norm(self.targetFaceV - check_face)
            print(ed)
            if ed < self.threshold:
                return True
            else:
                return False


    def compare(self):
        for f in listdir(self.input_folder):
            path = os.path.join(self.input_folder, f)
            # If current file is not a supported file type, skip it.
            if not os.path.isfile(path) or not accepted_file(f):
                continue
            print('scanning', f)
            # Finds faces in current image.
            try:
                faces = self.find_faces(path)
            except Exception as e:
                print("No faces found", e)
                cv2.imwrite(os.path.join(self.output_folder, f), cv2.imread(path))
                continue
                                                                           
            # Draws face - NEEDS TO BE BLUR - for every similar face found.
            im = None
            for i, face in enumerate(faces):
                v = np.array(face["embedding"])
                fa = face["facial_area"]
                # If face is not similar, skip it.
                if not self.is_similar(v) :
                    continue
                print('Found face')
                im = self.draw_face_rect(path, fa, img=im)
          
            # saves image.
            self.save_image(im, f, path)


    def save_image(self, image, name,  path):
        # Save image.
        if image is not None:
            cv2.imwrite(os.path.join(self.output_folder, name), image)
        # If no face was found, saves unaltered image.
        else:
            cv2.imwrite(os.path.join(self.output_folder, name), cv2.imread(path))
# --------------------------------------------------------

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-t','--target', help='Image path containing target face.', required=True)
    parser.add_argument('-i','--input', help='Folder path containing images to check against.', required=True)
    parser.add_argument('-o','--output', help='Name of the folder containing the output images.', default='ouput')
    parser.add_argument('-b','--blur', help='Blur mode: True to blur target face, False to blur all faces except target face. Default is True.', default=True)
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument('-c','--cosine', action='store_true',  help='Cosine similarity as face recognition mode.')
    group.add_argument('-e','--euclidean', action='store_true',  help='Euclidean distance as face recognition mode.')
    parser.add_argument('-th', '--threshold', help='Threshold that consideres a face similar enough used in the recognition mode. Default for cosine similarity is 0.65 & default for euclidean distance is 0.5.', type=float_range(0,1))
    args = vars(parser.parse_args())

    # Sets argument variables.
    global target_im, inp_folder, outp_folder, blur_mode, rec_mode
    target_im = args['target']
    inp_folder = args['input']
    outp_folder = args['output']
    if args['blur']:
        blur_mode = 'target'
    else:
        blur_mode = 'others'
    if args['cosine'] is not None:
        rec_mode = 'cosine'
    else:
        rec_mode = 'euclidean'
    if args['threshold'] is None:
        if rec_mode == 'cosine':
            threshold = 0.65
        else:
            threshold = 0.5
    else:
        threshold = args['threshold']
    
    # Creates output folder if this folder does not exist yet.
    if not os.path.exists(outp_folder):
        os.makedirs(outp_folder)

    recognizer = Recognizer(target_im, inp_folder, outp_folder, blur_mode, rec_mode, threshold) 
    recognizer.compare()

