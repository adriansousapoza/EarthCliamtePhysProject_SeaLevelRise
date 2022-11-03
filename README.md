# Earth and Climate Physics Project: Sea Level Rise

In order to run the file eacp_project.py and create_sampler.py you have to download this repository and change the variable 'datafolder' at the beginning of the python file to your own directory. For running the script you will need the folder data/raw_data from the repository https://github.com/cmip6moap/project01.

The file create_sampler.py creates the MCMC (emcee) samples and stores them as a .csv file (this usually takes a couple of minutes). This .csv file is used for analysis in the file eacp_project.py.
