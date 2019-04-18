# tractogram-cleaning
Computational methods for filtering out non-plausible streamlines from a tractogram
## Project workplan
https://docs.google.com/document/d/1HaMTnNh56o3Vw-3k3BEzfKY3mrm9wTGbvccZyH0MEHI/edit

## Done
* [x] Data inspection, report at [link]
* [x] Train PN with ModelNet40

## ToDo
* [ ] Prepare dataset P/NP streamlines for training
  * [ ] Retreive id streamlines P/NP in the original tracogram
  * [ ] Filter out streamlines shorter than 2.0cm
  * [ ] Generate b-splines from streamlines, using degree=3, and n=100 sample points
  * [ ] Add data to annex repository and clone it to Genova 
* [ ] Train PN with P/NP dataset

## Usage guide
insert here the list of commands needed to launch a training
