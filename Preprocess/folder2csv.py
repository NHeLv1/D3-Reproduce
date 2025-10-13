import os
import pandas as pd
from pandas import Series
from glob import glob

def count_images_in_folder(folder_path):
    image_names = [
        int(f.split('.')[0]) for f in os.listdir(folder_path)
        if f.lower().endswith(('.png', '.jpg', '.jpeg'))
    ]
    image_names.sort()
    return len(image_names), image_names


root_path = './GenVideo-100K/GenVideo-Val/GenVideo-I'
save_dir_prefix = './GenVideo-100K/GenVideo-Val'

records = {
    'content_path': [],
    'image_path': [],
    'type_id': [],
    'label': [],
    'frame_len': [],
    'frame_seq': []
}

# -----------------------------
# 处理 Fake 部分（两层目录）
# -----------------------------
fake_root = os.path.join(root_path, 'Fake')
for model_name in os.listdir(fake_root):
    model_dir = os.path.join(fake_root, model_name)
    if not os.path.isdir(model_dir):
        continue
    # 遍历每个视频子文件夹
    for video_name in os.listdir(model_dir):
        video_dir = os.path.join(model_dir, video_name)
        if not os.path.isdir(video_dir):
            continue
        frame_count, frame_seq = count_images_in_folder(video_dir)
        if frame_count == 0:
            continue

        # 取第一帧作为代表路径
        first_frame = sorted(glob(os.path.join(video_dir, '*')))[0]

        # 构造路径（用于 CSV）
        rel_content_path = os.path.relpath(video_dir, root_path)
        rel_frame_path = os.path.relpath(first_frame, root_path)
        content_path = os.path.join(save_dir_prefix, rel_content_path)
        frame_path = os.path.join(save_dir_prefix, rel_frame_path)

        records['content_path'].append(content_path)
        records['image_path'].append(frame_path)
        records['type_id'].append('AIGC视频')
        records['label'].append('1')
        records['frame_len'].append(frame_count)
        records['frame_seq'].append(frame_seq)


# -----------------------------
# 处理 Real 部分（一层目录）
# -----------------------------
real_root = os.path.join(root_path, 'Real')
for video_name in os.listdir(real_root):
    video_dir = os.path.join(real_root, video_name)
    if not os.path.isdir(video_dir):
        continue
    frame_count, frame_seq = count_images_in_folder(video_dir)
    if frame_count == 0:
        continue

    first_frame = sorted(glob(os.path.join(video_dir, '*')))[0]
    rel_content_path = os.path.relpath(video_dir, root_path)
    rel_frame_path = os.path.relpath(first_frame, root_path)
    content_path = os.path.join(save_dir_prefix, rel_content_path)
    frame_path = os.path.join(save_dir_prefix, rel_frame_path)

    records['content_path'].append(content_path)
    records['image_path'].append(frame_path)
    records['type_id'].append('真实视频')
    records['label'].append('0')
    records['frame_len'].append(frame_count)
    records['frame_seq'].append(frame_seq)


# -----------------------------
# 保存 CSV
# -----------------------------
df = pd.DataFrame(records)
df.to_csv('GenVideo.csv', encoding='utf-8', index=False)
print(f"✅ CSV generated: {len(df)} entries saved to GenVideo.csv")
