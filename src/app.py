import streamlit as st
import numpy as np
import matplotlib.pyplot as plt 
from PIL import Image
from streamlit_drawable_canvas import st_canvas
from pathlib import Path
import exemplar_based.inpainter as exemplar_inpainter
import fast_marching.inpainter as fm_inpainter
from mrf_inpainting.inpainter import MRFInpainting as MRF_Inpainter




def fast_marching_inpainter(img, mask_img):
    img = np.array(img)
    mask_img = np.array(mask_img)
    img_copy = img.copy()
    mask = mask_img[:, :, 0].astype(bool, copy=False) if len(mask_img.shape) == 3 else mask_img[:, :].astype(bool, copy=False)

    return fm_inpainter.inpaint(img_copy, mask)

def MRF_Result(img, mask):
    img = np.array(img)
    mask_img = np.array(mask)
    img_copy = img.copy()

    inpainter = MRF_Inpainter(patch_size=9, search_window=30, alpha=0.8, max_iterations=400)
    image, mask = inpainter.load_image_and_mask(input_img=img_copy, mask_img=mask_img)

    return inpainter.priority_inpaint()

    # return inpaint_image(img=img_copy, mask=mask_img)



LIGHT_BLUE = "rgba(29, 224, 202, 0.5)"
STROKE_WIDTH = 3
DRAWING_MODE = "polygon"
DEFAULT_IMAGE = "data/imgs/image4.png"
DEFAULT_SIZE = 256
DEFAULT_MASKS_DIR = Path("masks").resolve() # use absolutes
CANVAS_WIDTH = 500

st.title("Pexels - Image Inpainting 🌈")

st.sidebar.subheader("Image Inpainting Object Removal")
st.sidebar.text("Click the image to mask, right click to complete the mask. \n It is advisable to upload smaller images (256px x 256px) for faster processing ")

uploaded_image = st.sidebar.file_uploader("Input image:", type=["png", "jpg", "jpeg", "tiff", "tif"]) 

DRAWING_MODE = st.sidebar.selectbox(
            "Drawing tool:",
            ("freedraw", "polygon"),
        )
STROKE_WIDTH = st.sidebar.slider("Stroke width: ", 1, 25, 3)

# mask_dir = st.sidebar.text_input('Mask directory path:', DEFAULT_MASKS_DIR) # where masks will be saved
# if mask_dir != DEFAULT_MASKS_DIR:
#     if not os.path.exists(mask_dir):
#         st.sidebar.write(f"{mask_dir} is not a valid path!")

if uploaded_image: 
    image_to_annotate = Image.open(uploaded_image)
else:
    image_to_annotate = Image.open(DEFAULT_IMAGE)

## compute input image dimensions for later scalin
try:
    nx, ny, nd = np.array(image_to_annotate).shape
except: # 1band greyscale
    nx, ny = np.array(image_to_annotate).shape


# Otherwise draw the mask

canvas_result = st_canvas(
    fill_color = "rgba(232, 169, 21, 0.5)",
    stroke_color = LIGHT_BLUE,
    stroke_width = STROKE_WIDTH,
    background_image = image_to_annotate,
    height = int(CANVAS_WIDTH * float(nx/ny)) if uploaded_image else DEFAULT_SIZE,
    width = CANVAS_WIDTH if uploaded_image else DEFAULT_SIZE,
    drawing_mode = DRAWING_MODE, 
    display_toolbar = False,
)


if canvas_result.image_data is not None: # if there is annotation generate the mask
    mask_arr = canvas_result.image_data[:,:,0] # return first channel
    mask_arr = np.where(mask_arr > 0, 1, 0).astype(np.uint8) # binarise based on larger than zero

    mask_arr = mask_arr * 255 # convert to 0-255
    # st.image(mask_arr) # display the mask

    btn = st.button("Magic", icon="🌈", type="secondary")

    tab1, tab2, tab3 = st.tabs(["Exemplar-based", "Fast-Marching", "Markov-Random Field"])

    if btn:
        mask_img = Image.fromarray(mask_arr).convert('L')
        size = ny, nx
        mask_img = mask_img.resize(size)

        exemplar_based_result = exemplar_inpainter.Inpainter(np.array(image_to_annotate), np.array(mask_img), patch_size=9).inpaint()
        tab1.image(exemplar_based_result)

        fast_marching_result = fast_marching_inpainter(image_to_annotate, mask_img)
        tab2.image(fast_marching_result)

        mrf_result = MRF_Result(image_to_annotate, mask_img)
        tab3.image(mrf_result)

        # image_to_annotate.save("st/input.png")
        # mask_img.save("st/mask.png")

        #rescale images back to original size



