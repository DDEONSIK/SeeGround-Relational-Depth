import torch
import numpy as np
import matplotlib.pyplot as plt
from pytorch3d.structures import Pointclouds
from pytorch3d.renderer import (
    PointsRenderer,
    PointsRasterizationSettings,
    PointsRasterizer,
    AlphaCompositor,
    FoVPerspectiveCameras,
    PerspectiveCameras,
    look_at_view_transform,
)
from PIL import Image, ImageDraw, ImageFont
import os
import cv2
import json
import re
import random
from scipy.spatial import ConvexHull


def render_point_cloud_with_pytorch3d_with_objects(
    objects,
    targets,
    anchors,
    center,
    scan_pc,
    save_dir=None,
    image_size=680,
    use_color_image=True,
    draw_bbox=False,
    draw_id=False,
    draw_img=False,
    draw_mask=False,
    draw_contour=False,
    device="cuda",
    return_marker_coords=False, # 반환값 제어 플래그 추가
):
    """
    Render point cloud with PyTorch3D and annotate with objects, targets, and anchors.
    이미지 경로, 마커 좌표, 배경 마스크를 함께 반환.
    """
    point_cloud = create_point_cloud(scan_pc, device)
    os.makedirs(save_dir, exist_ok=True)

    # 앵커 존재 여부에 따라 시점의 기준점을 동적으로 결정
    if anchors:  # 앵커가 있으면, 앵커들의 평균 위치를 기준점으로 사용
        accumulated_positions = torch.zeros(3, dtype=torch.float32)
        for anchor in anchors:
            anchor_bbox_3d = torch.tensor(anchor["bbox_3d"][:3], dtype=torch.float32)
            accumulated_positions += anchor_bbox_3d
        look_at_point = accumulated_positions / len(anchors)
    else:  # 앵커가 없으면, 씬 전체의 중심을 기준점으로 사용 (중립적 시점)
        look_at_point = torch.tensor(center, dtype=torch.float32)


    cameras = setup_camera(
        anchor_bbox_3d=look_at_point, # 동적으로 결정된 기준점 전달
        center=center,
        image_size=image_size,
        camera_distance_factor=1,
        camera_lift=1.5,
        device=device,
        point_cloud=point_cloud,
        calibrate=False,
    )

    # render_point_cloud로부터 3개의 값(이미지, 마스크, 래스터라이저)을 모두 받음
    image_np, mask_np, rasterizer = render_point_cloud(point_cloud, cameras, image_size, device)
    depth_map = compute_depth_map(rasterizer, point_cloud)
    color_image = Image.fromarray((image_np * 255).astype(np.uint8))

    if not draw_img:
        width, height = color_image.size
        color_image = Image.new("RGB", (width, height), (255, 255, 255))

    font = ImageFont.truetype(
        # "/usr/share/fonts/truetype/freefont/FreeSans.ttf", 15, encoding="unic"
        # Dockerfile에 설치된 DejaVuSans.ttf로 변경
        "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 15, encoding="unic"
    )

    color = (128, 128, 128)
    
    # annotate_image 함수로부터 마커 좌표를 받음
    marker_coords_2d = annotate_image(
        color_image,
        anchors,
        targets,
        cameras,
        image_size,
        font,
        scan_pc=scan_pc,
        depth_map=None,
        bbox_color=color,
        draw_bbox=draw_bbox,
        draw_mask=draw_mask,
        draw_id=draw_id,
        draw_contour=draw_contour,
    )

    f_name = f"{save_dir}/rendered.png"
    color_image.save(f_name)
    print(f"Annotated image saved at {f_name}")
    color_image.close()

    # 플래그 값에 따라 반환값을 다르게 함. 반환값에 mask_np 추가
    if return_marker_coords:
        return f_name, marker_coords_2d, mask_np
    else:
        # 이 경우에도 일관성을 위해 3개의 값을 반환 (마커와 마스크는 비어있음)
        return f_name, {}, None


def annotate_image(
    color_image,
    anchors,
    targets,
    cameras,
    image_size,
    font,
    depth_map,
    scan_pc,
    bbox_color=(0, 255, 0),
    draw_bbox=False,
    draw_mask=False,
    draw_contour=False,
    draw_id=False,
):
    """
    Annotate the image and return the 2D coordinates of the drawn IDs.
    이제 마커 좌표 딕셔너리를 반환.
    """
    draw = ImageDraw.Draw(color_image, "RGBA")
    
    # 마커 좌표를 저장할 딕셔너리 초기화
    marker_coords_2d = {}

    if draw_mask:
        draw_masks(draw, targets, cameras, scan_pc, image_size)
    if draw_contour:
        draw_contours(draw, targets, cameras, scan_pc, image_size)
    if draw_bbox:
        draw_bboxes(draw, anchors + targets, cameras, image_size, bbox_color)
    if draw_id:
        # draw_ids 함수로부터 좌표 딕셔너리를 반환받음
        marker_coords_2d = draw_ids(draw, anchors + targets, cameras, image_size, font)

    return marker_coords_2d


# --- draw_masks, draw_contours, draw_bboxes 함수는 변경 없음 ---
def draw_masks(draw, targets, cameras, scan_pc, image_size):
    """
    Draw masks on the image.
    """
    for bbox in targets:
        bbox_id = bbox["bbox_id"]
        obj_label = bbox["label"]
        x, y, z, w, l, h = bbox["bbox_3d"]

        in_bbox_points = scan_pc[
            (scan_pc[:, 0] >= x - w / 2)
            & (scan_pc[:, 0] <= x + w / 2)
            & (scan_pc[:, 1] >= y - l / 2)
            & (scan_pc[:, 1] <= y + l / 2)
            & (scan_pc[:, 2] >= z - h / 2)
            & (scan_pc[:, 2] <= z + h / 2)
        ]

        projected_points = cameras.transform_points_screen(
            torch.tensor(in_bbox_points[:, :3]).cuda(),
            image_size=(image_size, image_size),
        )
        projected_points = projected_points[..., :2]

        visible_points = [(int(px), int(py)) for px, py in projected_points]

        mask_color = (
            random.randint(0, 255),
            random.randint(0, 255),
            random.randint(0, 255),
            100,
        )  # Random color with transparency
        draw.polygon(visible_points, fill=mask_color)


def draw_contours(draw, targets, cameras, scan_pc, image_size):
    """
    Draw contours on the image.
    """
    for bbox in targets:
        bbox_id = bbox["bbox_id"]
        obj_label = bbox["label"]
        x, y, z, w, l, h = bbox["bbox_3d"]

        in_bbox_points = scan_pc[
            (scan_pc[:, 0] >= x - w / 2)
            & (scan_pc[:, 0] <= x + w / 2)
            & (scan_pc[:, 1] >= y - l / 2)
            & (scan_pc[:, 1] <= y + l / 2)
            & (scan_pc[:, 2] >= z - h / 2)
            & (scan_pc[:, 2] <= z + h / 2)
        ]

        projected_points = cameras.transform_points_screen(
            torch.tensor(in_bbox_points[:, :3]).cuda(),
            image_size=(image_size, image_size),
        )
        projected_points = projected_points[..., :2]

        visible_points = [(int(px), int(py)) for px, py in projected_points]

        points_array = np.array(visible_points)
        try:
            hull = ConvexHull(points_array)  # Compute the convex hull
            contour_points = points_array[hull.vertices]  # Get contour points in order
            contour_points = [(int(x), int(y)) for x, y in contour_points]

            contour_color = (
                random.randint(0, 255),
                random.randint(0, 255),
                random.randint(0, 255),
                255,
            )  # Random color with transparency
            draw.line(
                contour_points + [contour_points[0]], fill=contour_color, width=3
            )  # Close the contour loop
        except:
            pass

def draw_bboxes(draw, bboxes, cameras, image_size, bbox_color):
    """
    Draw bounding boxes on the image.
    """
    for bbox in bboxes:
        x, y, z, w, l, h = bbox["bbox_3d"]
        corners = [
            [x - w / 2, y - l / 2, z - h / 2], [x - w / 2, y + l / 2, z - h / 2],
            [x + w / 2, y - l / 2, z - h / 2], [x + w / 2, y + l / 2, z - h / 2],
            [x - w / 2, y - l / 2, z + h / 2], [x - w / 2, y + l / 2, z + h / 2],
            [x + w / 2, y - l / 2, z + h / 2], [x + w / 2, y + l / 2, z + h / 2],
        ]
        corners_2d = cameras.transform_points_screen(
            torch.tensor(corners).cuda(), image_size=(image_size, image_size)
        )
        corners_2d = corners_2d[..., :2].cpu().numpy()
        valid_corners = [(0 <= px < image_size and 0 <= py < image_size) for px, py in corners_2d]
        if not any(valid_corners):
            continue
        draw_bbox_function(draw, corners_2d, valid_corners, bbox_color)

def draw_ids(draw, bboxes, cameras, image_size, font):
    """
    Draw object IDs on the image and return their 2D coordinates.
    이제 마커 좌표 딕셔너리를 반환.
    """
    # 마커 좌표를 저장할 딕셔너리 초기화
    marker_coords = {}
    
    for bbox in bboxes:
        bbox_id = bbox["bbox_id"]
        x, y, z, w, l, h = bbox["bbox_3d"]

        corners = [
            [x - w / 2, y - l / 2, z - h / 2], [x - w / 2, y + l / 2, z - h / 2],
            [x + w / 2, y - l / 2, z - h / 2], [x + w / 2, y + l / 2, z - h / 2],
            [x - w / 2, y - l / 2, z + h / 2], [x - w / 2, y + l / 2, z + h / 2],
            [x + w / 2, y - l / 2, z + h / 2], [x + w / 2, y + l / 2, z + h / 2],
        ]

        corners_2d = cameras.transform_points_screen(
            torch.tensor(corners).cuda(), image_size=(image_size, image_size)
        )
        corners_2d = corners_2d[..., :2].cpu().numpy()
        valid_corners = [(0 <= px < image_size and 0 <= py < image_size) for px, py in corners_2d]
        if not any(valid_corners):
            continue
            
        # draw_label 함수를 호출하고, 반환된 좌표를 저장
        marker_pos = draw_label(draw, corners_2d, bbox_id, font, image_size)
        if marker_pos:
            marker_coords[str(bbox_id)] = marker_pos
            
    # 최종 좌표 딕셔너리 반환
    return marker_coords


def draw_label(draw, corners_2d, bbox_id, font, image_size):
    """
    Draw label at the center of the top face and return its position.
    이제 마커의 중심 2D 좌표를 반환.
    """
    center_x = int((corners_2d[4][0] + corners_2d[5][0] + corners_2d[6][0] + corners_2d[7][0]) / 4)
    center_y = int((corners_2d[4][1] + corners_2d[5][1] + corners_2d[6][1] + corners_2d[7][1]) / 4)
    
    if 0 <= center_x < image_size and 0 <= center_y < image_size:
        text = f"{bbox_id}"
        try:
            # textbbox can fail with some fonts, handle this gracefully
            text_bbox = draw.textbbox((0, 0), text, font=font)
            text_width = text_bbox[2] - text_bbox[0]
            text_height = text_bbox[3] - text_bbox[1]
        except TypeError: # Fallback for older Pillow versions
            text_width, text_height = draw.textsize(text, font=font)

        background_x0 = center_x - text_width // 2 - 2
        background_y0 = center_y - text_height // 2 - 2
        background_x1 = center_x + text_width // 2 + 2
        background_y1 = center_y + text_height // 2 + 2
        
        draw.rectangle([background_x0, background_y0, background_x1, background_y1], fill=(255, 255, 255))
        draw.text(
            (center_x - text_width // 2, center_y - text_height // 2),
            text,
            font=font,
            fill=(255, 0, 0),
        )
        
        # ID가 그려진 중심 좌표를 반환
        return (center_x, center_y)
    
    # ID가 그려지지 않은 경우 None 반환
    return None

# --- draw_bbox_function, create_point_cloud, setup_camera, render_point_cloud, compute_depth_map 함수는 변경 없음 ---
def draw_bbox_function(draw, corners_2d, valid_corners, bbox_color):
    """
    Draw the 3D bounding box by connecting the projected corners.
    """
    for i, (start, end) in enumerate(
        [
            (0, 1), (1, 3), (3, 2), (2, 0), (4, 5), (5, 7),
            (7, 6), (6, 4), (0, 4), (1, 5), (2, 6), (3, 7),
        ]
    ):
        if valid_corners[start] and valid_corners[end]:
            draw.line(
                [tuple(corners_2d[start]), tuple(corners_2d[end])],
                fill=bbox_color,
                width=2,
            )
    return

def create_point_cloud(scan_pc, device):
    """
    Create a point cloud from scan data.
    렌더러가 RGBA를 출력하도록 포인트 클라우드 특징에 알파 채널(불투명도=1)을 추가.
    """
    points = torch.tensor(scan_pc[:, :3], dtype=torch.float32)
    colors = torch.tensor(scan_pc[:, 3:], dtype=torch.float32)
    
    # 알파 채널(불투명도)을 생성하고 기존 색상 채널과 결합
    alpha = torch.ones(colors.shape[0], 1, dtype=torch.float32)
    rgba = torch.cat([colors, alpha], dim=1)
    
    # RGBA 특징을 가진 포인트 클라우드 생성
    point_cloud = Pointclouds(points=[points], features=[rgba]).to(device)
    return point_cloud


def setup_camera(
    point_cloud, anchor_bbox_3d, center, image_size,
    camera_distance_factor=1.0, camera_lift=1.0, device="cuda", calibrate=True,
):
    """
    Set up the camera for rendering the point cloud.
    """
    center = torch.tensor(center, dtype=torch.float32)
    center[2] += camera_lift
    camera_position = center + camera_distance_factor * (center - anchor_bbox_3d)
    R, T = look_at_view_transform(
        dist=1, elev=0, azim=0, at=anchor_bbox_3d.unsqueeze(0),
        eye=camera_position.unsqueeze(0), up=((0, 0, 1),),
    )
    focal_length = torch.tensor([[1.0, 1.0]]).to(point_cloud.device)
    principal_point = torch.tensor([[0.0, 0.0]]).to(point_cloud.device)
    cameras = PerspectiveCameras(
        device=device, R=R, T=T, focal_length=focal_length, principal_point=principal_point,
    )
    if calibrate:
        if isinstance(image_size, int):
            image_size_tensor = torch.tensor([[image_size, image_size]])
        assert image_size_tensor.shape[-1] == 2
        points_2d = cameras.transform_points_screen(
            point_cloud.points_padded(), image_size=image_size_tensor
        )
        points_2d = points_2d[..., :2]
        min_proj = points_2d.min(dim=1)[0]
        max_proj = points_2d.max(dim=1)[0]
        new_focal_length = (
            focal_length * (max_proj - min_proj).max() / image_size_tensor.to(point_cloud.device)
        )
        new_principal_point = (min_proj + max_proj) / 2
        cameras = PerspectiveCameras(
            device=device, R=R, T=T, focal_length=new_focal_length,
            principal_point=new_principal_point,
        )
    return cameras


def render_point_cloud(point_cloud, cameras, image_size, device):
    """
    Render the point cloud and return the alpha mask.
    """
    # bin_size를 0으로 설정하여 네이브 래스터라이저를 사용.
    # 씬 전체를 렌더링할 때 발생하는 'bin overflow' 경고를 해결하고, 렌더링 정확성을 보장함.
    raster_settings = PointsRasterizationSettings(
        image_size=image_size, radius=0.01, points_per_pixel=10, bin_size=0
    )
    rasterizer = PointsRasterizer(cameras=cameras, raster_settings=raster_settings)
    
    # 배경색을 4채널 RGBA(0,0,0,0)로 설정. create_point_cloud에서 RGBA 특징을 생성했으므로 정상 작동
    renderer = PointsRenderer(
        rasterizer=rasterizer, compositor=AlphaCompositor(background_color=(0, 0, 0, 0))
    )
    
    images = renderer(point_cloud) # images.shape: (1, H, W, 4) -> RGBA
    
    # RGB 이미지 추출
    image_np = images[0, ..., :3].cpu().numpy()
    
    # 알파 채널(마스크)을 추출하여 2D 배열로 함께 반환
    mask_np = images[0, ..., 3].cpu().numpy()
    
    return image_np, mask_np, rasterizer


def compute_depth_map(rasterizer, point_cloud):
    """
    Compute the depth map of the point cloud.
    """
    fragments = rasterizer(point_cloud)
    depth_map = fragments.zbuf[0].cpu().numpy()
    depth_map = np.min(depth_map, axis=2)
    return depth_map

