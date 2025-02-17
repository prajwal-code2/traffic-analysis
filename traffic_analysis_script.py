import argparse
from typing import Dict, Iterable, List, Optional, Set

import cv2
import numpy as np
from tqdm import tqdm
from ultralytics import YOLO

import supervision as sv


COLORS = sv.ColorPalette.from_hex(["#E6194B", "#3CB44B", "#FFE119", "#3C76D1"])

ZONE_IN_POLYGONS = [
    np.array([[592, 282], [900, 282], [900, 82], [592, 82]]),
    np.array([[950, 860], [1250, 860], [1250, 1060], [950, 1060]]),
    np.array([[592, 582], [592, 860], [392, 860], [392, 582]]),
    np.array([[1250, 282], [1250, 530], [1450, 530], [1450, 282]]),
]

ZONE_OUT_POLYGONS = [
    np.array([[950, 282], [1250, 282], [1250, 82], [950, 82]]),
    np.array([[592, 860], [900, 860], [900, 1060], [592, 1060]]),
    np.array([[592, 282], [592, 550], [392, 550], [392, 282]]),
    np.array([[1250, 860], [1250, 560], [1450, 560], [1450, 860]]),
]



def initiate_polygon_zones(
    polygons: List[np.ndarray],
    triggering_anchors: Iterable[sv.Position] = [sv.Position.CENTER],
) -> List[sv.PolygonZone]:
    return [
        sv.PolygonZone(
            polygon=polygon,
            triggering_anchors=triggering_anchors,
        )
        for polygon in polygons
    ]


class DetectionsManager:
    def __init__(self) -> None:
        self.tracker_id_to_zone_id: Dict[int, int] = {}
        self.recorded_paths: Dict[int, Dict[int, Set]] = {}


    def update(
        self,
        detections: sv.Detections, 
        detections_zones_in: List[sv.Detections],
        detections_zones_out: List[sv.Detections]
    )-> sv.Detections:

        for zone_in_id, detections_zone_in in enumerate(detections_zones_in):
            for tracker_id in detections_zone_in.tracker_id:
                self.tracker_id_to_zone_id.setdefault(tracker_id, zone_in_id)
        
        for zone_out_id, detections_zone_out in enumerate(detections_zones_out):
            for tracker_id in detections_zone_out.tracker_id:
                if tracker_id in self.tracker_id_to_zone_id:
                    zone_in_id = self.tracker_id_to_zone_id[tracker_id]
                    self.recorded_paths.setdefault(zone_out_id, {})
                    self.recorded_paths[zone_out_id].setdefault(zone_in_id, set())
                    self.recorded_paths[zone_out_id][zone_in_id].add(tracker_id)

        detections.class_id = np.vectorize(
            lambda x: self.tracker_id_to_zone_id.get(x, -1))(detections.tracker_id
            )
        
        detections = detections[detections.class_id != -1]

        return detections


class VideoProcessor:
    def __init__(
        self,
        source_weights_path: str,
        source_video_path: str,
        target_video_path: Optional[str] = None,
        confidence_threshold: float = 0.3,
        iou_threshold: float = 0.7,
    ) -> None:
        self.conf_threshold = confidence_threshold
        self.iou_threshold = iou_threshold
        self.source_video_path = source_video_path
        self.target_video_path = target_video_path
        self.video_info = sv.VideoInfo.from_video_path(video_path=source_video_path)
        self.zones_in = initiate_polygon_zones(ZONE_IN_POLYGONS, [sv.Position.CENTER])
        self.zones_out = initiate_polygon_zones(ZONE_OUT_POLYGONS, [sv.Position.CENTER])
        self.detections_manager = DetectionsManager()
        self.tracer_annotator = sv.TraceAnnotator(color=COLORS, thickness=2, trace_length=100)

        self.model = YOLO(source_weights_path)
        self.box_annotator = sv.BoxAnnotator(color=COLORS)
        self.label_annotator = sv.LabelAnnotator(
            color=COLORS, text_color=sv.Color.BLACK
        )
        self.tracker = sv.ByteTrack()


    def process_video(self):
        frame_generator = sv.get_video_frames_generator(source_path=self.source_video_path)

        for frame in frame_generator:
            processed_frame = self.process_frame(frame=frame)
            cv2.imshow("frame",cv2.resize(processed_frame,(1280,720)))
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break

        cv2.destroyAllWindows()



    def annotate_frame(
            self, frame: np.ndarray, detections: sv.Detections
                       
        ) -> np.ndarray:
        annotated_frame = frame.copy()
        labels = [
            f"#{tracker_id}"
            for tracker_id in detections.tracker_id
        ]
        annotated_frame = self.box_annotator.annotate(
            scene=annotated_frame, 
            detections=detections,
            )
        annotated_frame = self.label_annotator.annotate(
            scene=annotated_frame, 
            detections=detections, 
            labels=labels
        )

        annotated_frame = self.tracer_annotator.annotate(
            scene=annotated_frame, 
            detections=detections
        )

        for i, (zone_in, zone_out) in enumerate(zip(self.zones_in, self.zones_out)):
            annotated_frame = sv.draw_polygon(
                scene=annotated_frame,
                polygon=zone_in.polygon,
                color=COLORS.colors[i],
            )

            annotated_frame = sv.draw_polygon(
                scene=annotated_frame,
                polygon=zone_out.polygon,
                color=COLORS.colors[i],
            )


        for zone_out_id, zone_out in enumerate(self.zones_out):
            zone_center = sv.get_polygon_center(polygon=zone_out.polygon)
            if zone_out_id in self.detections_manager.recorded_paths:
                paths = self.detections_manager.recorded_paths[zone_out_id]
                for i, zone_in_id in enumerate(paths):
                    count = len(paths[zone_in_id])
                    text_anchor = sv.Point(
                        x=zone_center.x,
                        y=zone_center.y+40*i,

                    )
                    annotated_frame = sv.draw_text(
                        scene=frame,
                        text=f"{count}",
                        text_anchor=text_anchor,
                        background_color=COLORS.colors[zone_in_id]
                    )

        return annotated_frame
    
    
    def process_frame(self, frame:np.ndarray)->np.ndarray:
        results = self.model(
            frame, verbose=False, conf=self.conf_threshold, iou=self.iou_threshold
            )[0]
        detections = sv.Detections.from_ultralytics(results)
        detections = self.tracker.update_with_detections(detections)

        detections_zones_in = []
        detections_zones_out = []

        for zone_in,zone_out in zip(self.zones_in,self.zones_out):
            detections_zone_in = detections[zone_in.trigger(detections=detections)]
            detections_zones_in.append(detections_zone_in)

            detections_zone_out = detections[zone_out.trigger(detections=detections)]
            detections_zones_out.append(detections_zone_out)

        
        detections = self.detections_manager.update(
            detections=detections, 
            detections_zones_in=detections_zones_in,
            detections_zones_out=detections_zones_out
            )
        

        return self.annotate_frame(frame, detections)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Traffic Flow Analysis with YOLO and ByteTracker"
    )

    parser.add_argument(
        "--source_weights_path",
        required=True,
        help="Path to the source weights file",
        type=str,
    )
    parser.add_argument(
        "--source_video_path",
        required=True,
        help="Path to the source video file",
        type=str,
    )
    parser.add_argument(
        "--target_video_path",
        default=None,
        help="Path to the target video file (output)",
        type=str,
    )
    parser.add_argument(
        "--confidence_threshold",
        default=0.3,
        help="Confidence threshold for the model",
        type=float,
    )
    parser.add_argument(
        "--iou_threshold", default=0.7, help="IOU threshold for the model", type=float
    )

    args = parser.parse_args()
    processor = VideoProcessor(
        source_weights_path=args.source_weights_path,
        source_video_path=args.source_video_path,
        target_video_path=args.target_video_path,
        confidence_threshold=args.confidence_threshold,
        iou_threshold=args.iou_threshold,
    )
    processor.process_video()        