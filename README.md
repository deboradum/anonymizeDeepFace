# anonymizeDeepFace

To gain some experience with my new Jetson Nano and Artificial Intelligence, I decided to create this program. When starting the program, a path to the image containing the target face needs to be provided, as well as the name of the folder where to images that are to be blurred are contained, and the name of the output folder. Then, using [Deepface](https://github.com/serengil/deepface), the target face vector is calculated. Next, for evert face detected in the input images that are to be blurred, either [cosine similarity](https://en.wikipedia.org/wiki/Cosine_similarity) or [euclidean distance](https://en.wikipedia.org/wiki/Euclidean_distance) is performed. When a face is detected to be similar enough. it is blurred. The program can also be configured to blur all faces *except* for the target face.

When a taret image with more than one face is given, the largest face is chosen as the target.

The program can be launched as follows:
```
python3 anonymize.py -t '<target_img_path>' -i '<input_folder>' [-o '<output_folder>'] [-b 'True'] (-c | -e) [-th <threshold_value>]
```
With:
```
-h, --help                  show this help message and exit
-t, --target TARGET         Image path containing target face.
-i, --input INPUT           Folder path containing images to check against.
-o, --output OUTPUT         Name of the folder containing the output images.
-b, --blur BLUR             Blur mode: True to blur target face, False to blur all faces except target face.
                            Default is True.
-c, --cosine                Cosine similarity as face recognition mode.
-e, --euclidean             Euclidean distance as face recognition mode.
-th, --threshold THRESHOLD  Threshold that consideres a face similar enough used in the recognition mode.
                            Default for cosine similarity is 0.65 & default for euclidean distance is 0.5.
```

Demo:

