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
    hide_st_style = """
                <style>
                #MainMenu {visibility:hidden;}
                footer > a {visibility:hidden; display:none}
                footer:after {content : "KRSBI HUMANOID"; color:White}
                """
    st.markdown(hide_st_style, unsafe_allow_html=True)
            
    col1, col2, col3 = st.columns(3)
    with col1 :
        SlideFilterUtama = ["Canny", "Threshold", "HSV", "YUV","BLOB"]
        listfilter = st.selectbox("Filter1", SlideFilterUtama)
        st.write(listfilter)
        if listfilter == "Canny":
            def video_frame_callback(frame):
                img = frame.to_ndarray(format="bgr24")
                img = cv2.cvtColor(cv2.Canny(img, 100, 200), cv2.COLOR_GRAY2BGR)
                return av.VideoFrame.from_ndarray(img, format="bgr24")

            webrtc_streamer(key="example", video_frame_callback=video_frame_callback,
                            media_stream_constraints={"video": True, "audio": False}, async_processing=True)
        elif listfilter == "Threshold":
            class VideoProcessor(VideoProcessorBase):
                def __init__(self):
                    self.threshold1 = 100
                    self.threshold2 = 200

                def recv(self, frame):
                    img = frame.to_ndarray(format="bgr24")
                    img = cv2.cvtColor(
                        cv2.Canny(img, self.threshold1, self.threshold2), cv2.COLOR_GRAY2BGR)
                    return av.VideoFrame.from_ndarray(img, format="bgr24")

            ths1 = webrtc_streamer(key="example", video_processor_factory=VideoProcessor,
                                  media_stream_constraints={"video": True, "audio": False})
            if ths1.video_processor:
                ths1.video_processor.threshold1 = st.slider("Threshold1", min_value=-255, max_value=255, step=1,
                                                           value=100)
                ths1.video_processor.threshold2 = st.slider("Threshold2", min_value=-255, max_value=255, step=1,
                                                           value=200)

        elif listfilter == "HSV":
            class VideoProcessor(VideoProcessorBase):
                def __init__(self):
                    self.lower_hsv1 = np.array([7, 66, 26])
                    self.upper_hsv2 = np.array([35, 239, 228])

                def recv(self,frame):
                    img = frame.to_ndarray(format="bgr24")
                    # img = cv2.inRange(img, lower_hsv1, upper_hsv2)
                    cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
                    # result = cv2.bitwise_and(img, img, img=img)
                    # img1 = cv2.inRange(img0, self.lower_hsv1, self.upper_hsv2, cv2.COLOR_GRAY2HSV)
                    return av.VideoFrame.from_ndarray(img, format="bgr24")

            hsv1 = webrtc_streamer(key="example", video_processor_factory=VideoProcessor,
                                  media_stream_constraints={"video": True, "audio": False})
            if hsv1.video_processor:
                hsv1.video_processor.lower_hsv1 = st.slider("Konfigurasi HSV 1", min_value=-255, max_value=255, step=1, value=100)
                hsv1.video_processor.upper_hsv2 = st.slider("Konfigurasi HSV 1", min_value=-255, max_value=255, step=1, value=150)

        elif listfilter == "YUV":
            class VideoProcessor(VideoProcessorBase):
                yuv1 = np.array([110, 50, 50])
                yuv2 = np.array([130, 255, 255])

                def recv(self, frame):
                    img = frame.to_ndarray(format="bgr24")
                    img = cv2.cvtColor(img, cv2.COLOR_BGR2YUV)
                    img = cv2.inRange(img, yuv1, yuv2)
                    return av.VideoFrame.from_ndarray(img1, format="bgr24")

            ctx = webrtc_streamer(key="example", video_processor_factory=VideoProcessor,
                                  media_stream_constraints={"video": True, "audio": False})
            if ctx.video_processor:
                ctx.video_processor.threshold1 = st.slider("Configure YUV1", min_value=-255, max_value=255, step=1,
                                                           value=100)
                ctx.video_processor.threshold2 = st.slider("Configure YUV1", min_value=-255, max_value=255, step=1,
                                                           value=150)
        elif listfilter == "BLOB":
            def video_frame_callback(frame):
                img = frame.to_ndarray(format="bgr24")
                #Coloring
                img_color = cv2.pyrDown(cv2.pyrDown(img))
                for _ in range(6):
                    img_color = cv2.bilateralFilter(img_color, 9, 9, 7)
                img_color = cv2.pyrUp(cv2.pyrUp(img_color))

                img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
                Thresh = cv2.adaptiveThreshold(img, 255, cv2.ADAPTIVE_THRESH_MEAN_C,
                                                       cv2.THRESH_BINARY_INV, 11, 2)
                img = cv2.cvtColor(Thresh, cv2.COLOR_GRAY2RGB)
                img = cv2.bitwise_and(img_color, img)
                return av.VideoFrame.from_ndarray(img, format="bgr24")

            webrtc_streamer(key="example", video_frame_callback=video_frame_callback,
                                    media_stream_constraints={"video": True, "audio": False}, async_processing=True)

    with col2 :
        SlideFilterUtama = ["Canny", "Threshold", "HSV", "YUV"]
        listfilter = st.selectbox("Filter2", SlideFilterUtama)
        st.write(listfilter)
        if listfilter == "Canny":
            def video_frame_callback(frame):
                img = frame.to_ndarray(format="bgr24")

                img = cv2.cvtColor(cv2.Canny(img, 100, 200), cv2.COLOR_GRAY2BGR)

                return av.VideoFrame.from_ndarray(img, format="bgr24")

            webrtc_streamer(key="example1", video_frame_callback=video_frame_callback,
                            media_stream_constraints={"video": True, "audio": False}, async_processing=True)
        elif listfilter == "Threshold":
            class VideoProcessor(VideoProcessorBase):
                def __init__(self):
                    self.threshold1 = 100
                    self.threshold2 = 200

                def recv(self, frame):
                    img = frame.to_ndarray(format="bgr24")
                    img = cv2.cvtColor(
                        cv2.Canny(img, self.threshold1, self.threshold2), cv2.COLOR_GRAY2BGR)
                    return av.VideoFrame.from_ndarray(img, format="bgr24")

            ctx = webrtc_streamer(key="example1", video_processor_factory=VideoProcessor,
                                  media_stream_constraints={"video": True, "audio": False})
            if ctx.video_processor:
                ctx.video_processor.threshold1 = st.slider("Configure Threshold2", min_value=-255, max_value=255, step=1,
                                                           value=100)
                ctx.video_processor.threshold2 = st.slider("Configure Threshold2", min_value=-255, max_value=255, step=1,
                                                           value=150)



        elif listfilter == "HSV":
            class VideoProcessor(VideoProcessorBase):
                # _, frame = webrtc_streamer.read()
                lower_hsv1 = np.array([7, 66, 26])
                upper_hsv2 = np.array([35, 239, 228])
                # orange_mask = cv2.inRange(img, low_red, high_red)
                # red = cv2.bitwise_and(frame, frame, mask=orange_mask)

                def recv(self, frame):
                    img = frame.to_ndarray(format="bgr24")
                    img = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
                    # img1 = cv2.inRange(img0, lower_hsv1, upper_hsv2)
                    return av.VideoFrame.from_ndarray(img, format="bgr24")

            ctx = webrtc_streamer(key="example1", video_processor_factory=VideoProcessor,
                                  media_stream_constraints={"video": True, "audio": False})
            if ctx.video_processor:
                ctx.video_processor.lower_hsv1 = st.slider("Konfigurasi HSV2", min_value=-255, max_value=255, step=1, value=100)
                ctx.video_processor.upper_hsv2 = st.slider("Konfigurasi HSV2", min_value=-255, max_value=255, step=1, value=150)

        elif listfilter == "YUV":
            class VideoProcessor(VideoProcessorBase):
                YUV2 = np.array([110, 50, 50])
                YUV2 = np.array([130, 255, 255])

                def recv(self, frame):
                    img = frame.to_ndarray(format="bgr24")
                    img = cv2.cvtColor(img, cv2.COLOR_BGR2YUV)
                    return av.VideoFrame.from_ndarray(img, format="bgr24")

            ctx = webrtc_streamer(key="example1", video_processor_factory=VideoProcessor,
                                  media_stream_constraints={"video": True, "audio": False})
            if ctx.video_processor:
                ctx.video_processor.YUV2 = st.slider("Konfigurasi YUV2", min_value=-255, max_value=255, step=1,
                                                           value=100)
                ctx.video_processor.YUV2 = st.slider("Konfigurasi YUV2", min_value=-255, max_value=255, step=1,
                                                           value=150)

    with col3 :
        SlideFilterUtama = ["Canny", "Threshold", "HSV", "YUV"]
        listfilter = st.selectbox("Filter3", SlideFilterUtama)
        st.write(listfilter)
        if listfilter == "Canny":
            def video_frame_callback(frame):
                img = frame.to_ndarray(format="bgr24")
                img = cv2.cvtColor(cv2.Canny(img, 100, 200), cv2.COLOR_GRAY2BGR)
                return av.VideoFrame.from_ndarray(img, format="bgr24")

            webrtc_streamer(key="example2", video_frame_callback=video_frame_callback,
                            media_stream_constraints={"video": True, "audio": False}, async_processing=True)
        elif listfilter == "Threshold":
            class VideoProcessor(VideoProcessorBase):
                def __init__(self):
                    self.threshold1 = 100
                    self.threshold2 = 200

                def recv(self, frame):
                    img = frame.to_ndarray(format="bgr24")
                    img = cv2.cvtColor(
                        cv2.Canny(img, self.threshold1, self.threshold2), cv2.COLOR_GRAY2BGR)
                    return av.VideoFrame.from_ndarray(img, format="bgr24")

            ctx = webrtc_streamer(key="example2", video_processor_factory=VideoProcessor,
                                  media_stream_constraints={"video": True, "audio": False})
            if ctx.video_processor:
                ctx.video_processor.threshold1 = st.slider("Configure Threshold3", min_value=-255, max_value=255, step=1,
                                                           value=100)
                ctx.video_processor.threshold2 = st.slider("Configure Threshold3", min_value=-255, max_value=255, step=1,
                                                           value=150)

        elif listfilter == "HSV":
            class VideoProcessor(VideoProcessorBase):
                lower_hsv1 = np.array([7, 66, 26])
                upper_hsv2 = np.array([35, 239, 228])

                def recv(self, frame):
                    img = frame.to_ndarray(format="bgr24")
                    img = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
                    # img1 = cv2.inRange(img0, lower_hsv1, upper_hsv2)

                    return av.VideoFrame.from_ndarray(img, format="bgr24")

            ctx = webrtc_streamer(key="example2", video_processor_factory=VideoProcessor,
                                  media_stream_constraints={"video": True, "audio": False})
            if ctx.video_processor:
                ctx.video_processor.lower_hsv1 = st.slider("Konfigurasi HSV3", min_value=-255, max_value=255, step=1, value=100)
                ctx.video_processor.upper_hsv2 = st.slider("Konfigurasi HSV3", min_value=-255, max_value=255, step=1, value=150)

        elif listfilter == "YUV":
            class VideoProcessor(VideoProcessorBase):
                YUV3 = np.array([110, 50, 50])
                YUV3 = np.array([130, 255, 255])

                def recv(self, frame):
                    img = frame.to_ndarray(format="bgr24")
                    img = cv2.cvtColor(img, cv2.COLOR_BGR2YUV)
                    return av.VideoFrame.from_ndarray(img, format="bgr24")

            ctx = webrtc_streamer(key="example2", video_processor_factory=VideoProcessor,
                                  media_stream_constraints={"video": True, "audio": False})
            if ctx.video_processor:
                ctx.video_processor.YUV3 = st.slider("Konfigurasi YUV3", min_value=-255, max_value=255, step=1,
                                                           value=100)
                ctx.video_processor.YUV3 = st.slider("Konfigurasi YUV3", min_value=-255, max_value=255, step=1,
                                                           value=150)
    # Button samping pertama dll dan konfigurasi!
    CobaPilih = ["Robot 1", "Robot 2","Robot 3"]
    choice = st.sidebar.selectbox("Pilih Robot", CobaPilih)
    CobaParameter = ["Walking Parameter", "Head Tracking Parameter"]
    param = st.sidebar.selectbox("Pilih Parameter", CobaParameter)

    if choice == "Robot 1":
        pilih = ["Gawang", "Bola", "Garis"]
        pilih_objek = st.sidebar.selectbox("Objek", pilih)
        input_angkaX = st.sidebar.number_input('X Move')
        st.sidebar.write('Angka Terakhir', input_angkaX)
        input_angkaY = st.sidebar.number_input('Y Move')
        st.sidebar.write('Angka Terakhir ', input_angkaY)
    elif choice == "Robot 2":
        pilih = ["Gawang", "Bola", "Garis"]
        pilih_objek = st.sidebar.selectbox("Objek", pilih)
        input_angkaX2 = st.sidebar.number_input('X Move')
        st.sidebar.write('Angka Terakhir ', input_angkaX2)
        input_angkaY2 = st.sidebar.number_input('Y Move')
        st.sidebar.write('Angka Terakhir ', input_angkaY2)
    elif choice == "Robot 3":
        pilih = ["Gawang", "Bola", "Garis"]
        pilih_objek = st.sidebar.selectbox("Objek", pilih)
        input_angkaX3 = st.sidebar.number_input('X Move')
        st.sidebar.write('Angka Terakhir ', input_angkaX3)
        input_angkaY3 = st.sidebar.number_input('Y Move')
        st.sidebar.write('Angka Terakhir ', input_angkaY3)

if __name__ == '__main__':
    main()
