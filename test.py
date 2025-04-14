import torch
import gc
import os
import cv2
import numpy as np
import sys
from pathlib import Path
import tempfile

sys.path.append("./sam2")

from sam2.build_sam import build_sam2_video_predictor
from ultralytics import YOLO

def extract_frames(video_path, temp_dir=None):
    """비디오에서 프레임 추출 후 임시 디렉토리에 저장"""
    if temp_dir is None:
        temp_dir = tempfile.mkdtemp(prefix="video_frames_")
    else:
        os.makedirs(temp_dir, exist_ok=True)
    
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise ValueError(f"비디오 파일을 열 수 없습니다: {video_path}")
    
    frame_count = 0
    frames = []
    frame_paths = []
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        frame_path = os.path.join(temp_dir, f"{frame_count}.jpg")
        cv2.imwrite(frame_path, frame)
        frames.append(frame)
        frame_paths.append(frame_path)
        frame_count += 1
    
    cap.release()
    print(f"총 {frame_count}개 프레임 추출 완료 ({temp_dir})")
    
    # 원본 비디오의 FPS 정보 가져오기
    video_fps = cap.get(cv2.CAP_PROP_FPS)
    if video_fps <= 0:  # FPS 정보를 얻을 수 없는 경우 기본값 사용
        video_fps = 30
    
    return frames, frame_paths, temp_dir, video_fps

def get_bbox_prompts(first_frame, detection_ckpt):
    """첫 프레임에서 바운딩 박스 추출"""
    # 임시 파일로 첫 번째 프레임 저장
    with tempfile.NamedTemporaryFile(suffix=".jpg", delete=False) as temp_file:
        temp_first_frame_path = temp_file.name
    
    cv2.imwrite(temp_first_frame_path, first_frame)
    
    detection_model = YOLO(detection_ckpt)
    detection_result = detection_model(temp_first_frame_path)[0]
    
    # 임시 파일 삭제
    os.unlink(temp_first_frame_path)

    bboxes = detection_result.boxes.xyxy
    bboxes_cpu = bboxes.to('cpu').numpy()
    
    # 클래스 정보 가져오기 (사람 클래스인지 확인)
    classes = detection_result.boxes.cls.cpu().numpy()
    
    prompts = {}
    
    # 사람 클래스(0)만 선택 또는 가장 큰 바운딩 박스 선택
    max_area = 0
    best_idx = 0
    
    for idx, (bbox, cls) in enumerate(zip(bboxes_cpu, classes)):
        area = (bbox[2] - bbox[0]) * (bbox[3] - bbox[1])
        if cls == 0 and area > max_area:  # 사람 클래스이면서 더 큰 바운딩 박스
            max_area = area
            best_idx = idx
    
    # 사람 클래스가 없는 경우, 가장 큰 객체 선택
    if max_area == 0 and len(bboxes_cpu) > 0:
        for idx, bbox in enumerate(bboxes_cpu):
            area = (bbox[2] - bbox[0]) * (bbox[3] - bbox[1])
            if area > max_area:
                max_area = area
                best_idx = idx
    
    # 적절한 바운딩 박스만 반환
    if len(bboxes_cpu) > 0:
        prompts[0] = ((int(bboxes_cpu[best_idx][0]), int(bboxes_cpu[best_idx][1]), 
                      int(bboxes_cpu[best_idx][2]), int(bboxes_cpu[best_idx][3])), 0)
        
    return prompts


if __name__ == "__main__":
    detection_ckpt = "ckpts/yolo11s.pt"
    samurai_cfg = "configs/samurai/sam2.1_hiera_s.yaml"
    samurai_ckpt = "sam2/checkpoints/sam2.1_hiera_small.pt"
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # MP4 파일 경로 지정
    video_path = "data/20231101_182105_1500429493682263.mp4"  # 입력 비디오 파일
    output_path = "result.mp4"     # 출력 비디오 파일
    
    # 비디오에서 프레임 추출
    loaded_frames, frame_paths, temp_frames_dir, video_fps = extract_frames(video_path)
    
    # SAM 예측기 초기화
    samurai_predictor = build_sam2_video_predictor(samurai_cfg, samurai_ckpt, device)
    prompts = get_bbox_prompts(loaded_frames[0], detection_ckpt)
    
    # 비디오 크기 정보 가져오기
    height, width = loaded_frames[0].shape[:2]
    
    # 결과 비디오 작성기 설정
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    output_video = cv2.VideoWriter(output_path, fourcc, video_fps, (width, height))
    
    with torch.inference_mode(), torch.autocast("cuda", dtype=torch.float16):
        # 초기 상태 설정
        initial_state = samurai_predictor.init_state(temp_frames_dir, offload_video_to_cpu=True)
        
        if not prompts:
            print("바운딩 박스를 찾을 수 없습니다.")
            # 모든 프레임을 그대로 출력
            for frame in loaded_frames:
                output_video.write(frame)
            output_video.release()
            sys.exit(0)
        
        bbox, tracking_label = prompts[0]
        
        print(f"선택된 바운딩 박스: {bbox}")
        
        # 첫 번째 프레임에 대한 마스크 초기화
        _, _, first_masks = samurai_predictor.add_new_points_or_box(initial_state, box=bbox, frame_idx=0, obj_id=0)
        
        # 모든 프레임에 대한 결과를 저장할 배열
        all_frame_results = [None] * len(loaded_frames)
        all_frame_results[0] = (bbox, first_masks[0][0].cpu().numpy() > 0.0)
        
        # 나머지 프레임 처리 (연속된 마스크 생성)
        for frame_idx, object_ids, masks in samurai_predictor.propagate_in_video(initial_state):
            if frame_idx < len(all_frame_results):
                # 첫 번째 객체(object_id=0)만 처리
                for i, object_id in enumerate(object_ids):
                    if object_id == 0:  # 우리가 관심있는 객체 ID
                        mask = masks[i][0].cpu().numpy() > 0.0
                        
                        # 바운딩 박스 계산
                        non_zero_indices = np.argwhere(mask)
                        if len(non_zero_indices) > 0:
                            y_min, x_min = non_zero_indices.min(axis=0).tolist()
                            y_max, x_max = non_zero_indices.max(axis=0).tolist()
                            box = [x_min, y_min, x_max, y_max]
                            
                            # 결과 저장
                            all_frame_results[frame_idx] = (box, mask)
                        else:
                            # 마스크가 비어있는 경우 이전 프레임의 결과 사용
                            if frame_idx > 0 and all_frame_results[frame_idx-1] is not None:
                                all_frame_results[frame_idx] = all_frame_results[frame_idx-1]
    
    # 비어있는 프레임 결과 채우기 (보간)
    last_valid_result = all_frame_results[0]
    for i in range(1, len(all_frame_results)):
        if all_frame_results[i] is None:
            all_frame_results[i] = last_valid_result
        else:
            last_valid_result = all_frame_results[i]
    
    # 결과 시각화 및 저장
    color = (0, 0, 255)  # 빨간색으로 변경
    
    for i, (frame, result) in enumerate(zip(loaded_frames, all_frame_results)):
        if result is not None:
            bbox, mask = result
            frame_copy = frame.copy()
            
            # 마스크 시각화
            mask_image = np.zeros((height, width, 3), dtype=np.uint8)
            mask_image[mask] = color
            frame_copy = cv2.addWeighted(frame_copy, 1, mask_image, 0.5, 0)
            
            # 바운딩 박스 그리기
            if len(bbox) == 4:
                cv2.rectangle(frame_copy, (bbox[0], bbox[1]), (bbox[2], bbox[3]), color, 2)
            
            output_video.write(frame_copy)
        else:
            output_video.write(frame)
    
    output_video.release()
    print(f"처리된 비디오가 {output_path}에 저장되었습니다.")
    
    # 임시 디렉터리 정리
    import shutil
    shutil.rmtree(temp_frames_dir, ignore_errors=True)
    
    del samurai_predictor, initial_state
    gc.collect()
    torch.clear_autocast_cache()
    torch.cuda.empty_cache()


