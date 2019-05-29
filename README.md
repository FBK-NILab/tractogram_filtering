# tractogram-cleaning
Computational methods for filtering out non-plausible streamlines from a tractogram
## Project workplan
https://docs.google.com/document/d/1HaMTnNh56o3Vw-3k3BEzfKY3mrm9wTGbvccZyH0MEHI/edit
## Project Results
https://docs.google.com/spreadsheets/d/1YDwBZPsPya5YHlQOk5_3vCnagxfnOr7BkXpA3nOrEyc/edit?usp=sharing

## Done
* [x] Data inspection, report at [link]
* [x] Train PN with ModelNet40
* [x] Prepare dataset P/NP streamlines for training
  * [x] Retreive id streamlines P/NP in the original tracogram
  * [x] Filter out streamlines shorter than 2.0cm
  * [x] Generate b-splines from streamlines, using degree=3, and n=100 sample points
  * [x] Add data to annex repository and clone it to Genova 
* [x] Train PN with P/NP dataset
* [x] Try random repetition dataset
* [x] Try zero padding dataset
* [x] Try frenet dataset

## ToDo
* [ ] Solve bug with model without T-Net
* [ ] Slides
* [ ] Report


## Usage guide
insert here the list of commands needed to launch a training
