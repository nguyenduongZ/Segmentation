import pathlib import Path

# Set up api path
f_path = Path.resolve(Path(__file__))
api_path = f_path.parent.parent

# Dataset_dir and path to files
dataset_dir = api_path.joinpath('/dataset/VGPhraseCut_v0')

img_path = dataset_dir.joinpath('images')
name_att_rel_count_fpath = dataset_dir.joinpath('name_att_rel_count.json')
image_data_split_fpath = dataset_dir.joinpath('/image_data_split.json')
skip_fpath = dataset_dir.joinpath('/skip.json')

refer_fpaths = dict()
refer_input_fpaths = dict()
scene_graphs_fpaths = dict()
# Split train, val, test, miniv
for split in ['train', 'val', 'test', 'miniv']:
    refer_fpaths[split] = dataset_dir.joinpath('refer_%s.json' % split)
    refer_input_fpaths[split] = dataset_dir.joinpath('refer_input_%s.json' % split)
    scene_graphs_fpaths[split] = dataset_dir.joinpath('scene_graphs_%s.json' % split)


