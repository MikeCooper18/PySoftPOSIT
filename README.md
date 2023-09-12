# Installation
- Open Anaconda enabled terminal.
- Create a new Anaconda environment with Python version `3.7`
- Install requirements using `pip install -r requirements.txt`
- Install the Meta SAM checkpoint [file](https://dl.fbaipublicfiles.com/segment_anything/sam_vit_h_4b8939.pth). 

# Usage
- Run `python main.py` to run the software.
- Select the whether or not to mask the image based upon the selected position. This masks the image using Meta's Segment-Anything model and is intended for use in which there are multiple objects in the image, to select the object to generate the key points for.
- Select the key point extraction method to use:
    - Vertex: Extracts the key points based upon vertex corner detection using the Harris corner detector model.
    - SIFT: Extracts the key points of the object based upon the SIFT model - not recommended.
- Select the image to extract the key points for via the file selector. The image will then be processed based on the settings.


# Not Implemented features
[] get keypoints of masked image using SIFT or similar.
    [] SIFT - Not perfect.
[] Get blender model vertex points
[] Pass to SoftPOSIT algorithm
[] Display results.
[] Make work for video iteratively.