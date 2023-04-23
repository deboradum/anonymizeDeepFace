from deepface import DeepFace
from pprint import pprint

input_folder = 'inps/'

def analyze_pic(path):
    embeddings = DeepFace.represent(img_path = path)

    print(len(embeddings))

    vec = embeddings[0]['embedding']
    face = embeddings[0]['facial_area']

    return vec, face


analyze_pic(f'{input_folder}pep1.JPG')
analyze_pic(f'{input_folder}pep_daanBril.JPG')
