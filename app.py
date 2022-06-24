import streamlit as st
import cv2
from PIL import Image, ImageEnhance
import numpy as np
import os
import av
from streamlit_webrtc import VideoProcessorBase, webrtc_streamer


@st.cache
def load_video(vid):
    vid1 = webrtc_streamer.open(vid)
    return vid1


face_cascade = cv2.CascadeClassifier(
    'frecog/haarcascade_frontalface_default.xml')
eye_cascade = cv2.CascadeClassifier('frecog/haarcascade_eye.xml')
smile_cascade = cv2.CascadeClassifier('frecog/haarcascade_smile.xml')




def main():
    """Detection App for Humanoid POLINEMA"""

    global our_image, result_eyes, image_file
    st.title("Image Processing Humanoid")
    st.text("Build with KRSBI-H \u2764\uFE0F by Hamba Allah")
    col1, col2, col3 = st.columns(3)
    with col1 :
        SlideFilterUtama = ["Canny", "Threshold", "HSV", "YUV"]
        listfilter = st.selectbox("Filter1", SlideFilterUtama)
        st.write(listfilter)
    with col2 :
        SlideFilterUtama = ["Canny", "Threshold", "HSV", "YUV"]
        listfilter = st.selectbox("Filter2", SlideFilterUtama)
        st.write(listfilter)
    with col3 :
        SlideFilterUtama = ["Canny", "Threshold", "HSV", "YUV"]
        listfilter = st.selectbox("Filter3", SlideFilterUtama)
        st.write(listfilter)


    #Canny
    if listfilter == "Canny":
        def video_frame_callback(frame):
                img = frame.to_ndarray(format="bgr24")

                img = cv2.cvtColor(cv2.Canny(img, 100, 200), cv2.COLOR_GRAY2BGR)

                return av.VideoFrame.from_ndarray(img, format="bgr24")

        webrtc_streamer(key="example",  video_frame_callback=video_frame_callback,media_stream_constraints={"video": True, "audio": False},async_processing=True)
    elif listfilter == "Threshold":
        class VideoProcessor(VideoProcessorBase):
            def __init__(self):
                self.threshold1 = 100
                self.threshold2 = 200
            def recv(self, frame):
                img = frame.to_ndarray(format="bgr24")
                img = cv2.cvtColor(
                    cv2.Canny(img, self.threshold1, self.threshold2), cv2.COLOR_GRAY2BGR
                )

                return av.VideoFrame.from_ndarray(img, format="bgr24")

        ctx = webrtc_streamer(key="example", video_processor_factory=VideoProcessor,media_stream_constraints={"video": True, "audio": False})
        if ctx.video_processor:
            ctx.video_processor.threshold1 = st.slider("Threshold1", min_value=-255, max_value=255, step=1, value=100)
            ctx.video_processor.threshold2 = st.slider("Threshold2", min_value=-255, max_value=255, step=1, value=200)



    elif listfilter == "HSV":
        class VideoProcessor(VideoProcessorBase):
            lower_hsv1 = np.array([7, 66, 26])
            upper_hsv2 = np.array([35, 239, 228])
            def recv(self, frame):
                img = frame.to_ndarray(format="bgr24")
                img = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
                # img1 = cv2.inRange(img0, lower_hsv1, upper_hsv2)

                return av.VideoFrame.from_ndarray(img, format="bgr24")

        ctx = webrtc_streamer(key="example", video_processor_factory=VideoProcessor,media_stream_constraints={"video": True, "audio": False})
        if ctx.video_processor:
            ctx.video_processor.hsv1 = st.slider("Konfigurasi", min_value=-255, max_value=255, step=1, value=100)
            ctx.video_processor.hsv2 = st.slider("Konfigurasi", min_value=-255, max_value=255, step=1, value=400)

    elif listfilter == "YUV":
        class VideoProcessor(VideoProcessorBase):
            def __init__(self):
                self.threshold1 = np.array([110, 50, 50])
                self.threshold2 = np.array([130, 255, 255])

            def recv(self, frame):
                img = frame.to_ndarray(format="bgr24")

                img = cv2.cvtColor(img, cv2.COLOR_BGR2YUV)

                return av.VideoFrame.from_ndarray(img, format="bgr24")

        ctx = webrtc_streamer(key="example", video_processor_factory=VideoProcessor,media_stream_constraints={"video": True, "audio": False})
        if ctx.video_processor:
            ctx.video_processor.threshold1 = st.slider("Threshold1", min_value=-255, max_value=255, step=1, value=100)
            ctx.video_processor.threshold2 = st.slider("Threshold2", min_value=-255, max_value=255, step=1, value=200)











    # Button samping pertama dll dan konfigurasi
    CobaPilih = ["Robot 1", "Robot 2","Robot 3"]
    choice = st.sidebar.selectbox("Pilih Robot", CobaPilih)

    if choice == "Robot 1":
        pilih = ["Gawang", "Bola", "Garis"]
        pilih_objek = st.sidebar.selectbox("Objek", pilih)
        input_angka = st.sidebar.number_input('X Move')
        st.sidebar.write('Angka Terakhir', input_angka)
    elif choice == "Robot 2":
        pilih = ["Gawang", "Bola", "Garis"]
        pilih_objek = st.sidebar.selectbox("Objek", pilih)
        input_angka = st.sidebar.number_input('X Move')
        st.sidebar.write('Angka Terakhir ', input_angka)
    elif choice == "Robot 3":
        pilih = ["Gawang", "Bola", "Garis"]
        pilih_objek = st.sidebar.selectbox("Objek", pilih)
        input_angka = st.sidebar.number_input('X Move')
        st.sidebar.write('Angka Terakhir ', input_angka)

    CobaParameter = ["Walking Parameter", "Head Tracking Parameter"]
    param = st.sidebar.selectbox("Pilih Parameter", CobaParameter)
    if param == ["Walking Parameter"] :
        Angka = st.sidebar.number_input('X Move')
        st.sidebar.write('Angka Terakhir ', Angka)
    elif param == ["HeadTracking Parameter"] :
        Angka = st.sidebar.number_input('Y Move')
        st.sidebar.write('Angka Terakhir ', Angka)




            # blur1 = ["Blur", "Median", "Bilateral"]
            # blur = st.sidebar.selectbox("Filter", blur1)
            #
            # if blur == 'Blur':
            #             y = st.sidebar.slider('Brightness',
            #                                   min_value=-0.0, max_value=255.0, step=0.1)
            #             check = st.sidebar.checkbox("Brightness")
            #             st.sidebar.write('Cek', check)
            #             print(y)
            #             st.sidebar.write("Jumlahnya:", y)
            #
            #             x = st.sidebar.slider('Contrast',
            #                                   min_value=-0.0, max_value=255.0, step=0.1)
            #             check1 = st.sidebar.checkbox("Contrast")
            #             st.sidebar.write('Cek', check1)
            #             print(x)
            #             st.sidebar.write("Jumlahnya:", x)
            #
            #             z = st.sidebar.slider('HSV',
            #                                   min_value=-0.0, max_value=255.0, step=0.1)
            #             check2 = st.sidebar.checkbox("HSV")
            #             st.sidebar.write('Cek', check2)
            #             print(z)
            #             st.sidebar.write("Jumlahnya:", z)
            #
            # elif blur == 'Median':
            #             y = st.sidebar.slider('Brightness',
            #                                   min_value=-0.0, max_value=255.0, step=0.1)
            #             check = st.sidebar.checkbox("Brightness")
            #             st.sidebar.write('Cek', check)
            #             print(y)
            #             st.sidebar.write("Jumlahnya:", y)
            #
            #             x = st.sidebar.slider('Contrast',
            #                                   min_value=0.0, max_value=255.0, step=0.1)
            #             check1 = st.sidebar.checkbox("Contrast")
            #             st.sidebar.write('Cek', check1)
            #             print(x)
            #             st.sidebar.write("Jumlahnya:", x)
            #
            #             z = st.sidebar.slider('HSV',
            #                                   min_value=0.0, max_value=255.0, step=0.1)
            #             check2 = st.sidebar.checkbox("HSV")
            #             st.sidebar.write('Cek', check2)
            #             print(z)
            #             st.sidebar.write("Jumlahnya:", z)
            #
            # elif blur == 'Bilateral':
            #             y = st.sidebar.slider('Brightness',
            #                                   min_value=0.0, max_value=255.0, step=0.1)
            #             check = st.sidebar.checkbox("Brightness")
            #             st.sidebar.write('Cek', check)
            #             print(y)
            #             st.sidebar.write("Jumlahnya:", y)
            #
            #             x = st.sidebar.slider('Contrast',
            #                                   min_value=0.0, max_value=255.0, step=0.1)
            #             check1 = st.sidebar.checkbox("Contrast")
            #             st.sidebar.write('Cek', check1)
            #             print(x)
            #             st.sidebar.write("Jumlahnya:", x)
            #
            #             z = st.sidebar.slider('HSV',
            #                                   min_value=0.0, max_value=255.0, step=0.1)
            #             check2 = st.sidebar.checkbox("HSV")
            #             st.sidebar.write('Cek', check2)
            #             print(z)
            #             st.sidebar.write("Jumlahnya:", z)

            # if choice == 'Image Processing':
            #     st.subheader("Image Processing")
            #
            #     image_file = st.file_uploader(
            #         "Upload Image", type=['jpg', 'png', 'jpeg'])
            # if image_file is not None:
            #         our_image = Image.open(image_file)
            #         st.text("Original Image")
            #         # st.write(type(our_image))
            #         st.image(our_image)
            #
            #         contoh_filter = st.sidebar.radio(
            # #             "Contoh Filter", ["Original", "Gray-Scale", "Contrast", "Brightness", "Blurring"])
            # if contoh_filter == 'Gray-Scale':
            #     img = np.array(img.convert('RGB'))
            #     vid3 = cv2.cvtColor(img, 1)
            #     gray = cv2.cvtColor(vid3, cv2.COLOR_BGR2GRAY)
            #     # st.write(new_img)
            #     st.image(gray)
            # elif contoh_filter == 'Contrast':
            #     c_rate = st.sidebar.slider("Contrast", 0.5, 3.5)
            #     enhancer = ImageEnhance.Contrast(img)
            #     img_output = enhancer.enhance(c_rate)
            #     st.image(img_output)
            #
            # elif contoh_filter == 'Brightness':
            #     c_rate = st.sidebar.slider("Brightness", 0.5, 3.5)
            #     enhancer = ImageEnhance.Brightness(img)
            #     img_output = enhancer.enhance(c_rate)
            #     st.image(img_output)
            #
            #         elif contoh_filter == 'Blurring':
            #             new_img = np.array(our_image.convert('RGB'))
            #             blur_rate = st.sidebar.slider("Blurring", 0.5, 3.5)
            #             img = cv2.cvtColor(new_img, 1)
            #             blur_img = cv2.GaussianBlur(img, (11, 11), blur_rate)
            #             st.image(blur_img)
            #         elif contoh_filter == 'Original':
            #
            #             st.image(our_image, width=300)
            #         else:
            #             st.image(our_image, width=300)

            #     # Image Detection
            #     task = ["Medium", "Gauss",
            #             "Bilateral", "Gray"]
            #     feature_choice = st.sidebar.selectbox("Find Features", task)
            #     if st.button("Process"):
            #
            #         if feature_choice == 'Medium':
            #             result_img, result_faces = detect_faces(our_image)
            #             st.image(result_img)
            #
            #             st.success("Found {} faces".format(len(result_faces)))
            #         elif feature_choice == 'Gauss':
            #             result_img = detect_smiles(our_image)
            #             st.image(result_img)
            #
            #         elif feature_choice == 'Bilateral':
            #             result_img, result_eyes = detect_eyes(our_image)
            #             st.success("Found {} Eyes".format(len(result_eyes)))
            #             st.image(result_img)
            #
            #         elif feature_choice == 'Gray':
            #             result_img = cartonize_image(our_image)
            #             st.image(result_img)
            #
            #         # elif feature_choice == 'Canonize':
            #         #     result_canny = cannize_image(our_image)
            #         #     st.image(result_canny)
            #
            #         # elif feature_choice == 'Eyes and Faces':
            #         #     result_img, result_faces = detect_faces(our_image)
            #         #     st.image(result_img)
            #         #     result_img = detect_eyes(our_image)
            #         #     st.image(result_img)
            #         #     st.success("Found {} faces and {} eyes".format(
            #         #         len(result_faces), len(result_eyes)))
            #



if __name__ == '__main__':
    main()
