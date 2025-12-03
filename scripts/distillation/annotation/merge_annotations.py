import os
from typing import List

class AnnotationMerger:
    def __init__(self):
        
        return
    def init_env(self, anno_type: str, sub_folders: List[str]):
        assert anno_type in ['train', 'eval'], "anno_type must be 'train' or 'eval'"
        self.sub_folders = sub_folders
        self.anno_type = anno_type
        self.accumulated_folder = '/media/levin/DATA/nerf/new_es8/stereo'
        self.accumulated_file = os.path.join(
            self.accumulated_folder,
            'picked_images_anno.txt' if anno_type == 'train' else 'picked_images_eval_anno.txt'
        )
        self.sub_file_name = (
            'picked_images_anno.txt' if anno_type == 'train' else 'picked_images_eval_anno.txt'
        )
        return

    def merge_annotations(self):
        merged_lines = []
        for sub_folder in self.sub_folders:
            anno_file = os.path.join(sub_folder, self.sub_file_name)
            print(f"Processing annotation file: {anno_file}")
            assert os.path.isfile(anno_file), f"Annotation file not found: {anno_file}"
            with open(anno_file, 'r', encoding='utf-8') as f:
                for line in f:
                    line = line.strip()
                    assert line, f"Empty line found in annotation file: {anno_file}"
                    left, right, disp = line.split()
                    # Convert to absolute paths
                    left_abs = os.path.normpath(os.path.join(sub_folder, left))
                    right_abs = os.path.normpath(os.path.join(sub_folder, right))
                    disp_abs = os.path.normpath(os.path.join(sub_folder, disp))
                    # Convert to relative paths to accumulated_folder
                    left_rel = os.path.relpath(left_abs, self.accumulated_folder)
                    right_rel = os.path.relpath(right_abs, self.accumulated_folder)
                    disp_rel = os.path.relpath(disp_abs, self.accumulated_folder)
                    merged_lines.append(f"{left_rel} {right_rel} {disp_rel}")
        # Save merged annotation file
        with open(self.accumulated_file, 'w', encoding='utf-8') as f:
            for line in merged_lines:
                f.write(line + '\n')
        print(f"Merged annotation file saved to {self.accumulated_file}")
    def run(self):
        # Hardcoded input as requested
        sub_folders = [
            '/media/levin/DATA/nerf/new_es8/stereo/20251119/1',
            '/media/levin/DATA/nerf/new_es8/stereo/20251119/2',
            '/media/levin/DATA/nerf/new_es8/stereo/20251119/3',
            '/media/levin/DATA/nerf/new_es8/stereo/20251119/4',
            '/media/levin/DATA/nerf/new_es8/stereo/20251119/5',
            '/media/levin/DATA/nerf/new_es8/stereo/20251119/6',
            '/media/levin/DATA/nerf/new_es8/stereo/20251119/7',
            '/media/levin/DATA/nerf/new_es8/stereo/20251119/8',
            '/media/levin/DATA/nerf/new_es8/stereo/20250702'

        ]
        anno_type = 'train'

        self.init_env(anno_type, sub_folders)
        self.merge_annotations()
        return

if __name__ == "__main__":
    merger = AnnotationMerger()
    merger.run()
