import streamlit as st

import os
import tensorflow as tf
from utils import load_image, show_n, get_image_download_link
import tensorflow_hub as hub

# Header
st.title("Neural Style Transfer")
st.write("Transfer your image based on the \"style\" of your other image. Based on the [Fast Style Transfer for Arbitrary Styles](https://www.tensorflow.org/hub/tutorials/tf2_arbitrary_image_stylization) tensorflow tutorial.")

# Parameters
output_image_size = 384
# The content image size can be arbitrary.
content_img_size = (output_image_size, output_image_size)
# The style prediction model was trained with image size 256 and it's the 
# recommended image size for the style image (though, other sizes work as 
# well but will lead to different results).
style_img_size = (256, 256)  # Recommended to keep it at 256.

MODEL_PATH = os.path.join(os.path.dirname(__file__), 'model')
# set cache dir to store hub module
os.environ['TFHUB_CACHE_DIR'] = MODEL_PATH
# turn off file_uploader warning
# TODO: wrap the file_uploader buffer with a IOWrapper
st.set_option('deprecation.showfileUploaderEncoding', False)

# Upload images
st.sidebar.title("Upload your images")
content_image_buffer = st.sidebar.file_uploader("Choose the source image", type=["png", "jpg", "jpeg"])
style_image_buffer = st.sidebar.file_uploader("Choose the style image", type=["png", "jpg", "jpeg"])

if (content_image_buffer is not None) and (style_image_buffer is not None):
    content_image = load_image(content_image_buffer, content_img_size)
    style_image = load_image(style_image_buffer, style_img_size)
    style_image = tf.nn.avg_pool(style_image, ksize=[3,3], strides=[1,1], padding='SAME')

    img_list = [content_image, style_image]
    name_list = ['Content image', 'Style image']
    st.pyplot(show_n(img_list, name_list))

    if st.button("Show me the style!"):

        with st.spinner("Styling your image now..."):
            # Load TF-Hub module.
            hub_handle = 'https://tfhub.dev/google/magenta/arbitrary-image-stylization-v1-256/2'
            hub_module = hub.load(hub_handle)

            # Stylize content image with given style image.
            outputs = hub_module(tf.constant(content_image), tf.constant(style_image))
            stylized_image = outputs[0]
        st.pyplot(show_n([stylized_image], ["Stylized image"]))
        # Add download button to sidebar
        st.sidebar.markdown(get_image_download_link(stylized_image), unsafe_allow_html=True)
        