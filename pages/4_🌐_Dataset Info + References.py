import streamlit as st 


st.markdown("""

## 

## Dataset Info

The models we used are [TraCE](https://www.cgd.ucar.edu/projects/trace), [FAMOUS](https://catalogue.ceda.ac.uk/uuid/77c4348fc49c4fcebfa201c6185a29fa), and [LOVECLIM]((https://www.elic.ucl.ac.be/modx/index.php?id=81) ) 

We parameterize lightning from paleoclimate model data using the function Lighting~=CONCLD x PRECIP (mm/day)

We considered time range from 21 kaBP to present We have converted each models time variable to the unit kaBP.  


## References

Community Climate System Model (CCSM3) and CESM1.2 (Liu et al., 2009; letter of collaboration: Bette Otto-Bliesner) \n
FAst Met Office/UK Universities Simulator (FAMOUS; Smith and Gregory, 2012) \n
Loch-Vecode-Ecbilt-Clio-Agism Model (LOVECLIM; Timm and Timmermann, 2007)\n 



""")