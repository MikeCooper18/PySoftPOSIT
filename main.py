import sys
import PyQt5.QtCore as QtCore
import PyQt5.QtGui as QtGui
import PyQt5.QtWidgets as qt
import cv2

import torch
import torchvision
import numpy as np
import matplotlib.pyplot as plt
from segment_anything import sam_model_registry, SamPredictor


# check if cuda is available
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

SAM_MODEL_TYPE = "vit_h"
SAM_MODEL_CHECKPOINT = "sam_vit_h_4b8939.pth"
sam = sam_model_registry[SAM_MODEL_TYPE](checkpoint = SAM_MODEL_CHECKPOINT)
sam.to(device = DEVICE)
predictor = SamPredictor(sam)

# Whether to use Meta SAM to make the image or not.
MASK_IMAGE = False

# The method used to get keypoints. "SIFT" (Scale Invariant Feature Transform) or "VERTEX" (Simple vertex detector)
KEYPOINT_METHOD = "VERTEX"


WIDTH = 1920
HEIGHT = 1080
PADDING = 20
BTN_HEIGHT = 50

image_path = None



# Functions for displaying masks
def show_mask(mask, ax, random_color=False):
    if random_color:
        color = np.concatenate([np.random.random(3), np.array([0.6])], axis=0)
    else:
        color = np.array([30/255, 144/255, 255/255, 0.6])
    h, w = mask.shape[-2:]
    mask_image = mask.reshape(h, w, 1) * color.reshape(1, 1, -1)
    ax.imshow(mask_image)

def show_points(coords, labels, ax, marker_size=375):
    pos_points = coords[labels==1]
    neg_points = coords[labels==0]
    ax.scatter(pos_points[:, 0], pos_points[:, 1], color='green', marker='*', s=marker_size, edgecolor='white', linewidth=1.25)
    ax.scatter(neg_points[:, 0], neg_points[:, 1], color='red', marker='*', s=marker_size, edgecolor='white', linewidth=1.25)

def show_box(box, ax):
    x0, y0 = box[0], box[1]
    w, h = box[2] - box[0], box[3] - box[1]
    ax.add_patch(plt.Rectangle((x0, y0), w, h, edgecolor='green', facecolor=(0,0,0,0), lw=2))

# Recenter keypoints so that the center of the image is (0, 0)
def recenterKeypoint(keypoint):
    recentered = (keypoint[0] - (WIDTH/2), keypoint[1] - (HEIGHT / 2))
    rounded = int(recentered[0]), int(recentered[1])
    return rounded


# Function which groups keypoints together which are less than a certain distance apart
def group_and_average_keypoints(keypoints, distance_threshold):
    # for each keypoint in the image, check if it is within the distance threshold of any of the other keypoints
    # if it is, then add it to the same group as that keypoint
    # if it is not, then create a new group for that keypoint
    keypoint_groups = [] # List of lists of keypoints
    for keypoint in keypoints:
        # Check if the keypoint is within the distance threshold of any of the other keypoints
        keypoint_added = False
        for group in keypoint_groups:
            for group_keypoint in group:
                # print(f"Group keypoint: {group_keypoint}")
                # print(f"Distance: {np.linalg.norm(np.array(keypoint) - np.array(group_keypoint))}")
                if np.linalg.norm(np.array(keypoint) - np.array(group_keypoint)) < distance_threshold:
                    # print("Adding keypoint to group")
                    group.append(keypoint)
                    keypoint_added = True
                    break
            if keypoint_added:
                break
        
        if not keypoint_added:
            # print("Creating new group")
            keypoint_groups.append([keypoint])
    

    # Calculate the average position of each keypoint group
    average_keypoint_groups = [] # List of tuples of keypoints
    for group in keypoint_groups:
        average_keypoint = (0, 0)
        for keypoint in group:
            average_keypoint = np.add(average_keypoint, keypoint)
        
        average_keypoint = np.divide(average_keypoint, len(group))
        average_keypoint = (int(average_keypoint[0]), int(average_keypoint[1]))
        average_keypoint_groups.append(average_keypoint)
    
    print(f"Num keypoints (grouped and averaged): {len(average_keypoint_groups)}")
    # print(f"Average keypoints: {average_keypoint_groups}")

    return average_keypoint_groups

def image_object_select_click(event, x, y, flags, param):
    if event == cv2.EVENT_LBUTTONDOWN:
        print(f"Clicked at pixel coordinates: ({x}, {y})")

        cv2.destroyAllWindows()


        masked_image = get_masked_image((x, y))

        if masked_image is not None:
            # cv2.imshow("Masked image", masked_image)

            keypoints, descriptors = get_keypoints(masked_image)
            keypoint_tuples = []
            if KEYPOINT_METHOD == "SIFT":
                # keypoints_tuple = [tuple([int(keypoint.pt[0]), int(keypoint.pt[1])]) for keypoint in keypoints]
                for keypoint in keypoints:
                    x = int(keypoint.pt[0])
                    y = int(keypoint.pt[1])
                    keypoint_tuples.append((x,y))
    
            elif KEYPOINT_METHOD == "VERTEX":
                # keypoints_tuple = [tuple([int(keypoint[1]), int(keypoint[0])]) for keypoint in zip(keypoints[0], keypoints[1])]
                for keypoint in zip(keypoints[0], keypoints[1]):
                    x = int(keypoint[1]) # TODO: why are the x and y coordinates flipped compared to SIFT?
                    y = int(keypoint[0])
                    keypoint_tuples.append((x,y))


            # print(f"Keypoints: {keypoints_tuple}")
            print(f"Num Keypoints (ungrouped): {len(keypoint_tuples)}")
            print(f"Keypoint tuples: {keypoint_tuples}")

            # Show image with circles on the keypoints
            image_with_keypoints = masked_image.copy()
            for keypoint in keypoint_tuples:
                # Make random colour for each keypoint
                colour = np.random.randint(0, 255, 3).tolist()

                cv2.circle(image_with_keypoints, keypoint, 3, colour, -1)
            
            # cv2.imshow("Image with keypoints", image_with_keypoints)
            # cv2.waitKey(0)


            # Group keypoints together which are less than a certain distance apart and average each group
            averaged_keypoints = group_and_average_keypoints(keypoint_tuples, 5)
            print(f"Averaged keypoints: {averaged_keypoints}")
            
            # Show image with circles on the average keypoints
            print("Key points:")
            image_with_average_keypoints = masked_image.copy()
            for keypoint in averaged_keypoints:
                colour = (0, 255, 0)

                radius = 7
                cv2.circle(image_with_average_keypoints, keypoint, radius, colour, -1)
                print(recenterKeypoint(keypoint))
            
            cv2.imshow("Image with average keypoints", image_with_average_keypoints)
            cv2.waitKey(0)
            cv2.destroyAllWindows()

# Masks image using Meta's SAM with an (x,y) coordinate as prompt input and returns the masked image. Only if MASK_IMAGE is set to True.
def get_masked_image(point: (int, int)):
    if image_path is None:
        print("Error: Invalid image path")
        print("image path: ", image_path)
        return
    
    image = cv2.imread(image_path)

    if MASK_IMAGE == False:
        return image

    if image is None:
        print("Error: image not found")
        return
    
    print("Image shape: ", image.shape)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    input_point = np.array([[point[0], point[1]]]) # Extract point to array.
    input_label = np.array([1]) # Set the point as a foreground point.

    print("Generating image embedding...")
    predictor.set_image(image)
    

    # Show the image with the selected point.
    # plt.figure(figsize=(10,10))
    # plt.imshow(image)
    # show_points(input_point, input_label, plt.gca())
    # plt.axis('on')
    # plt.show()

    print("Generating image masks...")
    # Call the predictor
    masks, scores, logits = predictor.predict(
        point_coords=input_point,
        point_labels=input_label,
        multimask_output=True,
    )

    print("Masks shape: ", masks.shape)

    # Show each of the different masks.
    if 0 :
        for i, (mask, score) in enumerate(zip(masks, scores)):
            plt.figure(figsize=(10,10))
            plt.imshow(image)
            show_mask(mask, plt.gca())
            show_points(input_point, input_label, plt.gca())
            plt.title(f"Mask {i+1} (score: {score:.3f})", fontsize=18)
            plt.axis("off")
            plt.show()


    best_mask = masks[np.argmax(scores), :, :]
    # Show the best mask
    if 0:
        plt.figure(figsize=(10,10))
        plt.imshow(image)
        show_mask(best_mask, plt.gca())
        show_points(input_point, input_label, plt.gca())
        plt.title(f"Best mask", fontsize=18)
        plt.axis("off")
        plt.show()

    print("Best mask before dilation: ", best_mask.shape)
    # print(best_mask)

    # Apply the mask to the image.
    masked_image = image.copy()
    masked_image[best_mask == 0] = 0

    return masked_image

def get_keypoints(image): # TODO: Need to adjust the threshold for the corner detector based on trial and error.
    if KEYPOINT_METHOD == "SIFT":
        # Read the selected image in and convert to grayscale.
        image_grayscale = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        # Initialise SIFT feature detector
        sift = cv2.SIFT_create()
        # Get image keypoints
        keypoints, descriptors = sift.detectAndCompute(image_grayscale, None)
        # Draw and show keypoints
        image_with_keypoints = cv2.drawKeypoints(image_grayscale, keypoints, None)
        cv2.imshow("Image with keypoints", image_with_keypoints)
        cv2.waitKey(0)

        return keypoints, descriptors

    elif KEYPOINT_METHOD == "VERTEX":
        image_grayscale = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        # Detect corners using the Harris corner detector
        corners = cv2.cornerHarris(image_grayscale, blockSize=2, ksize=3, k=0.04)
        # corners = cv2.dilate(corners, None)

        # Threshold and find coordinates of corner points
        threshold = None
        if MASK_IMAGE: # TODO: figure out threshold or change cornerHarris params so it works with masked images.
            threshold = 0.1 * corners.max()
            print(f"corners: {corners}")
            print(f"Threshold: {threshold}")
        else:
            threshold = 0.000003 * corners.max()
        
        corner_coordinates = np.where(corners > threshold)
        # print(f"Vertex coordinates: {corner_coordinates}")

        DRAW_CORNERS = False
        if DRAW_CORNERS:
            # Draw circles at corner points
            for x, y in zip(corner_coordinates[1], corner_coordinates[0]):
                cv2.circle(image, (x, y), 5, (0, 255, 0), -1)
            
            # Display the image
            cv2.imshow('Detected Vertices', image)
            cv2.waitKey(0)
            cv2.destroyAllWindows()
        
        return corner_coordinates, None




class MainWindow(qt.QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("SoftPOSIT+")
        self.setGeometry(100, 100, WIDTH, HEIGHT)
        self.setFixedSize(WIDTH, HEIGHT)
        
        palette = QtGui.QPalette()  # Create a palette
        palette.setColor(QtGui.QPalette.Window, QtGui.QColor("lightGray"))  # Set window background color
        self.setPalette(palette)
        
        # Not yet implemented - doesn't do anything with the file path bar display it.
        open_object_button = qt.QPushButton("Open Blender File", self)
        open_object_button.setGeometry(PADDING, PADDING, 200, 50)
        open_object_button.clicked.connect(self.open_blender_file_dialog)

        # Make label for selected file
        self.label_selected_file = qt.QLabel(self)
        self.label_selected_file.setGeometry(PADDING, 80, WIDTH - PADDING - PADDING, 2 * PADDING)
        self.label_selected_file.setText("No file selected")

        # Make button to open image file
        open_image_button = qt.QPushButton("Open Image File", self)
        open_image_button.setGeometry(PADDING + PADDING + 200, PADDING, 200, 50)
        open_image_button.clicked.connect(self.open_image_file_dialog)

        # Make label for selected image
        self.label_selected_image = qt.QLabel(self)
        self.label_selected_image.setGeometry(PADDING, 120, WIDTH - PADDING - PADDING, 2 * PADDING)
        self.label_selected_image.setText("No image selected")

        # Checkbox to choose if SAM masking is on
        self.checkbox_mask_image = qt.QCheckBox("Mask Image", self)
        self.checkbox_mask_image.setGeometry(PADDING, 160, 200, 50)
        self.checkbox_mask_image.setChecked(MASK_IMAGE)
        self.checkbox_mask_image.stateChanged.connect(self.checkbox_mask_image_changed)

        # Dropdown to select keypoint method
        self.dropdown_keypoint_method = qt.QComboBox(self)
        self.dropdown_keypoint_method.setGeometry(PADDING + 200, 200, 200, 50)
        self.dropdown_keypoint_method.addItems(["SIFT", "VERTEX"])
        self.dropdown_keypoint_method.setCurrentText(KEYPOINT_METHOD)
        self.dropdown_keypoint_method.currentTextChanged.connect(self.dropdown_keypoint_method_changed)

        # Label for keypoint method dropdown
        self.label_keypoint_method = qt.QLabel(self)
        self.label_keypoint_method.setGeometry(PADDING, 200, 200, 50)
        self.label_keypoint_method.setText("Keypoint Method:")



    def open_blender_file_dialog(self):
        options = qt.QFileDialog.Options()
        options |= qt.QFileDialog.ReadOnly
        
        file_path, _ = qt.QFileDialog.getOpenFileName(self, "Open Blender File", "", "Blender Files (*.blend)", options=options)
        
        if file_path:
            self.label_selected_file.setText(file_path)
    
    def open_image_file_dialog(self):
        options = qt.QFileDialog.Options()
        options |= qt.QFileDialog.ReadOnly
        
        file_path, _ = qt.QFileDialog.getOpenFileName(self, "Open Image File", "", "Image Files (*.png *.jpg *.jpeg)", options=options)
        
        if file_path:
            self.label_selected_image.setText(file_path)

            global image_path
            image_path = file_path

            # Show the image in a new window
            cv2.imshow("Image", cv2.imread(image_path))
            cv2.setMouseCallback("Image", image_object_select_click)
    
    def keyPressEvent(self, event):
        if event.key() == QtCore.Qt.Key_Escape:
            cv2.destroyAllWindows()            
            self.close()  # Close the application when Escape key is pressed
    
    def checkbox_mask_image_changed(self, state):
        global MASK_IMAGE
        MASK_IMAGE = state == QtCore.Qt.Checked
        print("Mask image: ", MASK_IMAGE)

    def dropdown_keypoint_method_changed(self, text):
        global KEYPOINT_METHOD
        KEYPOINT_METHOD = text
        print("Keypoint method: ", KEYPOINT_METHOD)



def main():

    print("PyTorch version:", torch.__version__)
    print("Torchvision version:", torchvision.__version__)
    print("Device: ", DEVICE)

    app = qt.QApplication(sys.argv)
    window = MainWindow()
    window.show()
    sys.exit(app.exec_())

if __name__ == "__main__":
    main()
