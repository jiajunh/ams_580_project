# ams_580_project

## Environment
* Using python3.12.6, all the required packages are in the 'requirements.txt' file

  ```
  pip install -r requirements.txt
  ```
* 

## proprocesing
* Three cols has " ?", not only "?", is missing values, so for every col, if the data type is str, first strip it.
* ‘education’ column and ‘education-num’ column is identical, so actually we can just neglect one, I choose to drop ‘education’