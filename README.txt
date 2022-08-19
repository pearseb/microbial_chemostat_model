Welcome to the 0D chemostat model!

Authors: Emily Zakem & Pearse Buchanan


The following is a brief descriptions of what the different scripts are, but refer to the scripts
themselves for more information...


 - model.py
The main model code. Contains time-loop and equations to run the chemostat (i.e. Sources and sinks, etc.)


 - call_model_*.py
This is how we call the model.py and provide the initial conditions and traits of microbes.


 - traits_new.py
 - traits_old.py
These are files containing information about the traits of the microbes involved in the chemostat.


 - yield_from_stoichiometry.py
Contains equation to calculate the yield of heterotrophy from Sinsabaugh et al. (2013). 
Called by "traits_new.py"


- diffusive_o2_coefficient.py
Contains equation to calculate the diffusive rate of O2 transport into cells


- O2_star.py
Contains equation to calculate subsistence O2 concentration for a microbe (O2*)


- R_star.py
Contains equation to calculate subsistence substrate concentration for a microbe (R*)


- line_plot.py
Code to automatically plot the time-series outcomes of the major model outputs
