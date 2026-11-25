# Tractogram filtering using Geometric Deep Learning

<!-- brainlife-app:begin (generated from the App manifest; edit the manifest, not this block) -->

> **Filters anatomically non-plausible streamlines from a .trk tractogram with a geometric deep learning classifier (Astolfi et al., MICCAI 2020), after warping to MNI space with ANTs, and outputs the cleaned tractogram and streamline indexes.**

Resamples each streamline to 16 points, registers the subject to MNI space using the T1w or FA image (fast SyN by default) and classifies every streamline as anatomically plausible or not with a geometric deep learning network trained on the Petit et al. 2019 definitions. Use it to remove artifactual streamlines from a whole-brain tractogram before bundle segmentation or connectomics. If neither T1w nor FA is given, the tractogram must already be in MNI space. Runs the GPU image of the tool (singularity --nv).

| | |
|---|---|
| 📥 **Inputs** | `neuro/track/trk` · `neuro/anat/t1w` (optional) · `neuro/tensor` (optional) |
| 📤 **Outputs** | `neuro/track/trk` · `raw` ×2 |
| 🐳 **Container** | `pietroastolfi/tractogram-filtering:gpu` |
| ⚙️ **Resources** | 1 CPUs · 0.5 h walltime |
| 🔗 **Run it** | [doi.org/10.25663/brainlife.app.390](https://doi.org/10.25663/brainlife.app.390) |

#### 📥 Inputs

| Input | Datatype | Required | Description |
|---|---|---|---|
| `1` | `neuro/track/trk` | yes | Input tractogram in trk format. |
| `2` | `neuro/anat/t1w` | no | T1w in dwi space. This argument is not required if tractogram is already in the MNI space. |
| `3` | `neuro/tensor` | no | FA image of the subject. This argument is not required if tractogram is already in the MNI space. |

#### 📤 Outputs

| Output | Datatype | Description |
|---|---|---|
| `output` | `neuro/track/trk` | A tractogram where artifactual streamlines has been filtered out. The output is in .trk format. |
| `output/plausible_idxs` | `raw` | Txt file containing the indexes of anatomically plausible streamline with respect to the input tractogram. |
| `output/non_plausible_idxs` | `raw` | Txt file containing the indexes of anatomically non-plausible streamline with respect to the input tractogram. |

<details>
<summary><b>⚙️ Parameters</b> (1)</summary>

| Parameter | Type | Default | Description |
|---|---|---|---|
| `fast_warp` | boolean | `true` | This parameter determine the speed of the non linear registration to standard MNI. When set to true it triggers a fast (few mins) but decent non-linear warp computation (search antsRegistrationSynQuick.sh for details). When set to false it triggers a slower (~30mins) but refined non linear registration. |

</details>

📚 **References used by this App:**
- Astolfi P. et al. Tractogram Filtering of Anatomically Non-plausible Fibers with Geometric Deep Learning. MICCAI 2020, LNCS 12267.
- Avesani P. et al. The open diffusion data derivatives, brain data upcycling via integrated publishing of derivatives and reproducible open cloud services. Sci Data 6, 69 (2019). [doi:10.1038/s41597-019-0073-y](https://doi.org/10.1038/s41597-019-0073-y)

<!-- brainlife-app:end -->

Computational methods for filtering out non-plausible streamlines from a tractogram

## Container details
The [built Docker image](https://hub.docker.com/repository/docker/pietroastolfi/tractogram-filtering) contains all the required packages to run the filtering in inference mode. The repository containes two dockers tagged as `cpu` and `gpu`, which run the tool using the cpu or the gpu. The container can be launched either using Docker, which requires the administrator privileges and some commands to mount external directories, or using Singularity (suggested), which instead runs without privileges and automatically mounts the home directory of the localhost (important for data retreival). 

In my tests I adopted the latest stable version of both Docker (19.03.8) and Singularity (3.5.3).

## Tractogram filtering script

The executable script (still under development) is `tractogram_filtering.py`. It reads the configuration file, `run_config.json` to get arguments from "outside", and based on it performs different steps.

The script generates a temporary folder `TEMP=tmp_tractogram_filtering/`, where it stores in the subdirectories `TEMP/input/` and `TEMP/output/` the actual input and output files. Some intermediate files generated during the pre-processing steps are stored directly in the `TEMP` folder 

The input file is always a tractogram .trk, projected into MNI space with fixed number of points per streamline. 

The output are two text files containing the indexes of plausible and non-plausible fibers, and optionally the .trk of the filtered tractogram.    

## Configuration file
`run_config.json` is composed as follows:
- `trk`: path to the tractogram uploaded by the user
- `t1`: path to the t1 image in subject space. The image is preferred if it is a brain extracted image. In case no t1 image is provided, the tractogram is assumed to be already in MNI space.
- `resample_points`: T/F flag. If T the streamlines will be resampled to 16 points, otherwise no.
- `return_trk`: T/F flag. If T the filtered trk tractogram will be returned along with the indexes of plausible and non-plausible streamlines.
- `task`: classification/regression. [not used right now]

## Usage
1. Create a json config file, using the one in the repo as example. In the repo inside `data/` I included a t1 and a small tractogram(.trk) that can be used for tests.
2. launch the container using the desired `<tag>=cpu | gpu`. In case you use `gpu` also use the `--nv` option with `singularity exec` or `--gpus all` with `docker run`:
    * using Singularity: from a writable directory launch the following command:\
  `singularity exec -e [--nv] docker://pietroastolfi/tractogram-filtering:<tag> tractogram_filtering.py -config <path-to-json>`
    <!-- - `$ sudo docker run --name tract_filtering -it pietroastolfi/tractogram-filtering:<tag> bash`\
    `$ sudo docker exec docker://pietroastolfi/tractogram-filtering-cpu "tractogram_filtering.py -config <path-to-json>"` -->
    * using Docker: `[sudo] docker run -v /home/<user>:/home/<user> --name tract_filtering [--gpus all] -it pietroastolfi/tractogram-filtering:<tag> bash`

To launch a shell inside the docker the command is `singularity shell -e docker://pietroastolfi/tractogram-filtering-cpu`
