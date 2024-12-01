# atlas-final

Below is my working Docker Container for the Higgs boson detection ATLAS experiment. 

To run, execute these two lines:

docker build -t my-analysis-container .

docker run --rm -it -v $(pwd)/output:/app/output my-analysis-container


The program can be scaled by adding datasets (e.g. data_A, data_F) to the process.py file.
The variable fraction can also be changed to edit the amount of data passed through, to speed up the process.
