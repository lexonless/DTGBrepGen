# Data Process Examples

## 1. Minimal example

Use your own STEP directory, keep the default parsed-data location:

```bash
python -m data_process.brep_process \
  --input /path/to/step_root \
  --option Fusion360
```

If one STEP file may contain multiple solids and you want to save them separately:

```bash
python -m data_process.brep_process \
  --input /path/to/step_root \
  --option Fusion360 \
  --split_multi_solid
```

This writes processed `.pkl` files to:

```bash
data_process/GeomDatasets/fusion360_parsed
```

Generate the train/val/test split:

```bash
python -m data_process.deduplicate \
  --name cad \
  --data data_process/GeomDatasets/fusion360_parsed \
  --option Fusion360
```

This writes the split file to:

```bash
data_process/fusion360_data_split_6bit.pkl
```

Generate VAE training data:

```bash
python -m data_process.deduplicate \
  --name facEdge \
  --data data_process/GeomDatasets/fusion360_parsed \
  --option Fusion360 \
  --list data_process/fusion360_data_split_6bit.pkl \
  --type face

python -m data_process.deduplicate \
  --name facEdge \
  --data data_process/GeomDatasets/fusion360_parsed \
  --option Fusion360 \
  --list data_process/fusion360_data_split_6bit.pkl \
  --type edge
```

Generate topology data:

```bash
python -c "from topology.datasets import create_topo_datasets; create_topo_datasets('train', 'Fusion360'); create_topo_datasets('test', 'Fusion360')"
```


## 2. Custom output locations

Write parsed geometry data to any directory:

```bash
python -m data_process.brep_process \
  --input /path/to/step_root \
  --option ABC \
  --output /path/to/processed/abc_parsed
```

Write the split file to any directory:

```bash
python -m data_process.deduplicate \
  --name cad \
  --data /path/to/processed/abc_parsed \
  --option ABC \
  --save_path /path/to/splits/abc_data_split_6bit.pkl
```

Write face/edge VAE data to any directory:

```bash
python -m data_process.deduplicate \
  --name facEdge \
  --data /path/to/processed/abc_parsed \
  --option ABC \
  --list /path/to/splits/abc_data_split_6bit.pkl \
  --type face \
  --save_path /path/to/splits/abc_data_split_6bit_face.pkl

python -m data_process.deduplicate \
  --name facEdge \
  --data /path/to/processed/abc_parsed \
  --option ABC \
  --list /path/to/splits/abc_data_split_6bit.pkl \
  --type edge \
  --save_path /path/to/splits/abc_data_split_6bit_edge.pkl
```

Write topology data to any directory:

```bash
python -c "from topology.datasets import create_topo_datasets; create_topo_datasets('train', 'ABC', geom_root='/path/to/processed/abc_parsed', split_path='/path/to/splits/abc_data_split_6bit.pkl', topo_root='/path/to/topo/abc'); create_topo_datasets('test', 'ABC', geom_root='/path/to/processed/abc_parsed', split_path='/path/to/splits/abc_data_split_6bit.pkl', topo_root='/path/to/topo/abc')"
```


## 3. Training examples

Geometry training:

```bash
python -m geometry.train_geom --name Fusion360 --option faceBbox

python -m geometry.train_geom \
  --name ABC \
  --data_root /path/to/processed/abc_parsed \
  --train_list /path/to/splits/abc_data_split_6bit.pkl \
  --option faceGeom
```

Topology training:

```bash
python -m topology.train_topo --name Fusion360 --option faceEdge

python -m topology.train_topo \
  --name ABC \
  --data_root /path/to/topo/abc \
  --option edgeVert
```


## 4. Notes

- `--option` is now just the dataset source name. It can be `ABC`, `Fusion360`, `MyDataset`, etc.
- The code normalizes that name internally, so `ABC` becomes `abc`, `Fusion360` becomes `fusion360`.
- If `config.yaml` does not contain a matching config section, training falls back to the `custom` config.
- `brep_process.py` now prints a summary of failure reasons such as `multi_solid_skipped_files`, `parse_filtered`, `bspline_failed`, and `save_exception`.
