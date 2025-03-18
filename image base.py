import argparse
import os
import supervision as sv
from ultralytics import YOLO
from tqdm import tqdm


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--source",
        type=str,
        required=True,
        help="Path to the input video"
    )
    parser.add_argument(
        "--checkpoint",
        type=str,
        default="yolov8l-seg.pt",
        help="Path or ID of the model checkpoint"
    )
    parser.add_argument(
        "--names",
        nargs="+",
        default=["cow"],
        help="List of class names to set for the model"
    )
    parser.add_argument(
        "--output",
        type=str,
        help="Path to save the annotated video"
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda",
        help="Device to run inference on (e.g., 'cpu' or 'cuda')"
    )
    return parser.parse_args()


def main():
    args = parse_args()

    # Check if the source file exists
    if not os.path.exists(args.source):
        raise FileNotFoundError(f"Source file '{args.source}' does not exist.")

    # Set default output path if not provided
    if not args.output:
        base, ext = os.path.splitext(args.source)
        args.output = f"{base}-output{ext}"

    # Load the YOLO model
    model = YOLO(args.checkpoint)
    model.to(args.device)

    # Get the model's class names
    model_class_names = model.names

    # Check if the provided names are valid
    for name in args.names:
        if name not in model_class_names.values():
            raise ValueError(f"Class name '{name}' is not valid. Valid class names are: {list(model_class_names.values())}")

    # Initialize video frame generator
    frame_generator = sv.get_video_frames_generator(args.source)
    video_info = sv.VideoInfo.from_video_path(args.source)

    # Initialize video sink to save the annotated video
    with sv.VideoSink(args.output, video_info) as sink:
        for frame in tqdm(frame_generator, total=video_info.total_frames, desc="Processing Video"):
            # Run inference on the frame
            results = model.predict(frame, verbose=False)

            # Convert results to Detections object
            detections = sv.Detections.from_ultralytics(results[0])

            # Calculate annotation parameters
            resolution_wh = (frame.shape[1], frame.shape[0])  # (width, height)
            thickness = sv.calculate_optimal_line_thickness(resolution_wh=resolution_wh)
            text_scale = sv.calculate_optimal_text_scale(resolution_wh=resolution_wh)

            # Generate labels
            labels = []
            for class_id, confidence in zip(detections.class_id, detections.confidence):
                class_name = model_class_names[class_id]  # Get class name from model's class names
                if class_name in args.names:  # Only include classes in args.names
                    labels.append(f"{class_name} {confidence:.2f}")
                else:
                    labels.append("")  # Add an empty label for classes not in args.names

            # Annotate the frame
            annotated_frame = frame.copy()
            annotated_frame = sv.MaskAnnotator(
                color_lookup=sv.ColorLookup.INDEX,
                opacity=0.4
            ).annotate(scene=annotated_frame, detections=detections)
            annotated_frame = sv.BoxAnnotator(
                color_lookup=sv.ColorLookup.INDEX,
                thickness=thickness
            ).annotate(scene=annotated_frame, detections=detections)
            annotated_frame = sv.LabelAnnotator(
                color_lookup=sv.ColorLookup.INDEX,
                text_scale=text_scale,
                smart_position=True
            ).annotate(scene=annotated_frame, detections=detections, labels=labels)

            # Write the annotated frame to the output video
            sink.write_frame(annotated_frame)

    print(f"Annotated video saved to: {args.output}")


if __name__ == "__main__":
    main()