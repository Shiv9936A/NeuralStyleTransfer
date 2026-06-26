from moviepy.editor import VideoFileClip
# from moviepy import VideoFileClip

# INPUT_VIDEO = "test/input.mp4"
# OUTPUT_VIDEO = "outputs/output.mp4"

def merge_audio(original_video_path,styled_video_path,final_output):

    styled_video = VideoFileClip(styled_video_path)
    original_video = VideoFileClip(original_video_path)

    audio = original_video.audio

    final_video = styled_video.set_audio(audio)

    final_video.write_videofile(
        final_output,
        codec="libx264",
        audio_codec="aac",
    )

    return final_output