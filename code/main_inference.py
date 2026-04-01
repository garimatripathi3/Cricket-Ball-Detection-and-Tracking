import config
from utils import gather_videos
from detector import BallDetector
from pipeline import process_video


def main():
    detector = BallDetector(
        model_path=config.MODEL_PATH,
        ball_class_id=config.BALL_CLASS_ID,
        conf=config.CONF,
        iou=config.IOU,
        imgsz=config.IMGSZ,
        device=config.DEVICE
    )

    videos = gather_videos(config.INPUT_PATH)
    if not videos:
        print(f"no videos found in {config.INPUT_PATH}")
        return

    for video_path in videos:
        print(f"\nprocessing {video_path}")
        process_video(video_path, detector)


if __name__ == "__main__":
    main()
