# 1. 把 STEP 预处理成 pkl
python -m data_process.brep_process --input E:\your_step_folder --option custom

# 2. 生成 train/val/test split
python -m data_process.deduplicate --name cad --data data_process/GeomDatasets/custom_parsed --option custom

# 3. 生成 face/edge VAE 训练集
python -m data_process.deduplicate --name facEdge --data data_process/GeomDatasets/custom_parsed --option custom --list data_process/custom_data_split_6bit.pkl --type face
python -m data_process.deduplicate --name facEdge --data data_process/GeomDatasets/custom_parsed --option custom --list data_process/custom_data_split_6bit.pkl --type edge

# 4. 生成 topology 数据
python -c "from topology.datasets import create_topo_datasets; create_topo_datasets('train', 'custom'); create_topo_datasets('test', 'custom')"

# 5. 训练
python -m geometry.train_vae --data data_process/GeomDatasets/custom_parsed --train_list data_process/custom_data_split_6bit_face.pkl --val_list data_process/custom_data_split_6bit.pkl --env custom_vae_face --option face
python -m geometry.train_vae --data data_process/GeomDatasets/custom_parsed --train_list data_process/custom_data_split_6bit_edge.pkl --val_list data_process/custom_data_split_6bit.pkl --env custom_vae_edge --option edge

python -m geometry.train_geom --name custom --option faceBbox
python -m geometry.train_geom --name custom --option vertGeom
python -m geometry.train_geom --name custom --option edgeGeom
python -m geometry.train_geom --name custom --option faceGeom

python -m topology.train_topo --name custom --option faceEdge
python -m topology.train_topo --name custom --option edgeVert