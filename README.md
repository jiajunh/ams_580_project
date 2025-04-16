# ams_580_project

## Environment
* Using python3.12.6, all the required packages are in the 'requirements.txt' file

* Suggest to use conda ot python virtual environment. If you are using conda
  ```
  conda create -n "ams580" python=3.12.6
  ```
	To activate the conda virtual environment
	
	```
  conda activate ams580
  ```
* Go to the project dirctory, and install all the requires packages
  ```
  pip install -r requirements.txt
  ```

* To run the final job, stay at the project directory and run the scripts/train.sh file. And then you should see it's running and output the results.
  ```
  ./scripts/train.sh
  ```


## proprocesing

* Three cols has " ?", not only "?", is missing values, so for every col, if the data type is str, first strip it.
* ‘education’ column and ‘education-num’ column is identical, so actually we can just neglect one, I choose to drop ‘education-num’

* ‘fnlwgt’ has a very biased distribution, maybe use log scale is better
* ‘capital gain / capital loss’ has large range, so need to scale down, use log
* native-country & race is highly biased, so maybe think to lower the
* Can merge race to white & not white
* Age to bin, each bin has similar percentage
* education merge based on diploma
* Marital merge into 3
* Work class can merge ‘never-worked’ & ‘without-pay’
* country merge to America & non-amerada