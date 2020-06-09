# Tractogram filtering using Geometric Deep Learning
Computational methods for filtering out non-plausible streamlines from a tractogram

## Container details
The [built Docker image](https://hub.docker.com/repository/docker/pietroastolfi/tractogram-filtering-cpu) contains all the required packages to run the filtering in inference mode using only CPU. The container can be launched either using Docker, which requires the administrator privileges and some commands to mount external directories, or using Singularity (suggested), which instead runs without privileges and automatically mounts the home directory of the localhost (important for data retreival). 

In my tests I adopted the latest stable version of both Docker (19.03.8) and Singularity (3.5.3).

## Tractogram filtering script

The executable script (still under development) is `tractogram_filtering.py`. It reads the configuration file, `run_config.json` to get arguments from "outside", and based on it performs different steps.

The script generates a temporary folder `TEMP=tmp_tractogram_filtering/`, where it stores in the subdirectories `TEMP/input/` and `TEMP/output/` the actual input and output files. Some intermediate files generated during the pre-processing steps are stored directly in the `TEMP` folder 

The input file is always a tractogram .trk, projected into MNI space with fixed number of points per streamline. 

The output are two text files containing the indexes of plausible and non-plausible fibers, and optionally the .trk of the filtered tractogram.    

## Configuration file
`run_config.json` is composed as follows:
- `trk`: path to the tractogram uploaded by the user>
- `t1`: path to the t1 image in subject space. The image is preferred if it is a brain extracted image. In case no t1 image is provided, the tractogram is assumed to be already in MNI space.
- `resample_points`: T/F flag. If T the streamlines will be resampled to 16 points, otherwise no.
- `return_trk`: T/F flag. If T the filtered trk tractogram will be returned along with the indexes of plausible and non-plausible streamlines.
- `task`: classification/regression. [not used right now]

## Usage
1. Create a json config file, using the one in the repo as example. In the repo inside `data/` I included a t1 and a small tractogram(.trk) that can be used for tests.
2. From a writable directory launch the following command:\
  `singularity exec -e docker://pietroastolfi/tractogram-filtering-cpu tractogram_filtering.py -config <path-to-json>`
    <!-- - `$ sudo docker run --name tract_filtering -it pietroastolfi/tractogram-filtering-cpu bash`\
    `$ sudo docker exec docker://pietroastolfi/tractogram-filtering-cpu "tractogram_filtering.py -config <path-to-json>"` -->

To launch a shell inside the docker the command is `singularity shell -e docker://pietroastolfi/tractogram-filtering-cpu`
