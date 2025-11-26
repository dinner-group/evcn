Before running any scripts, specify the locations of your raw topology and trajectory files in `data/files.json` as follows:
```
{
    "topology": topology_file,
    "trajectories": [
        [trajectory_file_1.1, trajectory_file_1.2, ...],
        [trajectory_file_2.1, trajectory_file_2.2, ...],
        ...
    ]
}
```
* `topology_file` is the path to the topology file.
* Each trajectory may be split across multiple files. `trajectory_file_i.j` is the path to the trajectory file containing segment `j` of trajectory `i`.

Run the scripts in the following order:
```
./preprocess.py
./train.sh
./validate.sh
./empirical.sh
./plot.py
```
