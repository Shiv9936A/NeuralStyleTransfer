print("APP STARTED")

import streamlit as st
import torch
import os
import zipfile
from PIL import Image
from io import BytesIO              # download btn needs bytes not PILImg
from streamlit_image_comparison import image_comparison
from infer import stylize_image
from video import stylize_video
from audio_editor import merge_audio
from training_config import TRAINING_CONFIG
from plot_training import generate_loss_graph

print("IMPORTS FINISHED")

STYLE_NAMES = list(
    TRAINING_CONFIG.keys()
)
DEVICE = torch.device(
    "cuda" if torch.cuda.is_available()
    else "cpu"
)

print("STREAMLIT READY")

st.set_page_config(
    page_title="Neural Style Transfer",
    layout="wide"
)

st.sidebar.title("Settings")
st.sidebar.subheader("Model Information")

image_size = st.sidebar.slider(
    "Image Size",
    128,
    512,
    256,
    step=64
)

st.title(
    "🎨 Neural_Style Transfer"
)

st.write(
    "Upload an image and apply the style"
)

image_tab, video_tab, batch_tab, analytics_tab = st.tabs(
    ["🖼 Image Stylization","🎥 Video Stylization","📂 Batch Inference","📊 Analytics"]
)

with image_tab:
    style = st.selectbox(
        "Choose style",
        STYLE_NAMES
    )

    strength = st.slider(
        "Style Strength",
        min_value=0.0,
        max_value=1.0,
        value=1.0,
        step=0.1
    )

    st.write(
        f"Selected Style: {style}"
    )

    uploaded_file = st.file_uploader(
        "Upload File",
        type=["jpg","jpeg","png"]
    )

    generate = st.button("Generate")
    if uploaded_file:
        image = Image.open(
            uploaded_file
        )
        st.image(
            image,
            caption="Uploaded Image",
            width=250
        )


        if generate and uploaded_file:
            model_path = TRAINING_CONFIG[style]["save_path"]

            progress = st.progress(0)
            with st.spinner(
                "Generating..."
            ):
                progress.progress(20)
                try:
                    output_image = stylize_image(image,model_path,DEVICE,strength)
                except Exception as e:
                    st.error(f"Error: {e}")
                    st.stop()

                progress.progress(100)
                st.success("Image Generated Successfully!")

                buffer = BytesIO()
                output_image.save(
                    buffer,
                    format="PNG"
                )
                buffer.seek(0)

                # col1, col2 = st.columns(2)
                # with col1:
                #     st.image(
                #         image,
                #         caption="Original"
                #     )
                # with col2:
                #     st.image(
                #         output_image,
                #         caption="Stylized"
                #     )

                st.subheader("Before / After Comparison")

                image_comparison(
                    img1=image,
                    img2=output_image,
                    label1="Original",
                    label2="Stylized",
                    width=500
                )

                st.download_button(
                    label="Download Image",
                    data=buffer,
                    file_name=f"{style}_output.png",
                    mime="image/png"                    # this tells the browser that it is an image
                )


with video_tab:
    video_style = st.selectbox(
        "Choose Video Style",
        STYLE_NAMES,
        key = "video_style"
    )

    uploaded_video = st.file_uploader(
        "Upload Video",
        type=["mp4"],
        key="video_upload"
    )

    if uploaded_video:
        input_video_path = "temp/input.mp4"
        output_video_path = "temp/output.mp4"
        final_video_path = "temp/final_output.mp4"
        model_path = TRAINING_CONFIG[video_style]["save_path"]

        st.video(uploaded_video)

        video_generate = st.button(
            "Generate Video",
            key="video_generate"
        )

        if video_generate and uploaded_video:
            os.makedirs(
                "temp",
                exist_ok=True
            )

            with open(input_video_path, "wb") as f:
                f.write(uploaded_video.read())

            with st.spinner("Stylizing video.."):
                try:
                    stats = stylize_video(input_video_path,output_video_path,model_path,DEVICE)
                    merge_audio(input_video_path,output_video_path,final_video_path)
                except Exception as e:
                    st.error(f"Error: {e}")
                    st.stop()

            st.success("Video Generated Successfully")

            col1,col2,col3 = st.columns(3)
            with col1:
                st.metric(
                    "Frames",
                    stats["frames"]
                )
            with col2:
                st.metric(
                    "Input FPS",
                    round(stats["fps"],2)
                )
            with col3:
                st.metric(
                    "Processing FPS",
                    round(stats["processing_fps"],2)
                )

            st.video(final_video_path)

            with open(final_video_path,"rb") as file:
                st.download_button(
                    label="Download Video",
                    data=file,
                    file_name=f"{video_style}_video.mp4",
                    mime="video/mp4"
                )

                
with batch_tab:
    batch_style = st.selectbox(
        "Choose Style",
        STYLE_NAMES,
        key="batch_style"
    )

    batch_files = st.file_uploader(
        "Upload Multiple Images",
        type=["jpg","jpeg","png"],
        accept_multiple_files=True
    )

    batch_generate = st.button("Generate Batch")

    if batch_generate and batch_files:
        model_path = TRAINING_CONFIG[batch_style]["save_path"]

        zip_buffer = BytesIO()
        with zipfile.ZipFile(zip_buffer,"w") as zip_file:
            for file in batch_files:
                image = Image.open(file)
                output = stylize_image(image,model_path,DEVICE,1.0)
                image_buffer = BytesIO()
                output.save(image_buffer,format="PNG")
                image_buffer.seek(0)
                zip_file.writestr(
                    f"styled_{file.name}",
                    image_buffer.read()
                )
                st.image(output,caption=file.name)

        zip_buffer.seek(0)

        st.download_button(
            label="📦 Download All Outputs",
            data=zip_buffer,
            file_name="styled_outputs.zip",
            mime="application/zip"
        )


with analytics_tab:
    graph_style = st.selectbox(
        "Select Style",
        STYLE_NAMES,
        key="graph_style"
    )

    graph_file = (f"loss_history_{graph_style}.json")

    if os.path.exists(graph_file):
        graph_path = generate_loss_graph(graph_style)

        st.image(
            graph_path,
            caption="Training Loss Curve"
        )

    else:
        st.warning(
            "No training history found."
        )

