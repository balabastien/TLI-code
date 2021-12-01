# TLI-code
Code for the TLI paper arXiv:2110.14651


The repository contains the codes that were used to produce the four panels of Fig 4; in particular, it contains




1. A code for computing the winding number, plotted on Fig. 4.b, contained in the file "winding_number_TLI.py"
2. A code for computing the surface Chern number in z-direction, plotted on Fig. 4.c, contained in the file "chern_number_z.py"
3. A code for computing the surface Chern number in y-direction, plotted on Fig. 4.d, contained in the file "chern_number_y.py"
4. A code for computing the average level spacing ratio, plotted on Fig. 4.a, contained in the file "level_stat_TLI.py". To run this file for a given system size, one needs to generate beforehand a unitary matrix that is generated with the file "unitarmatrixgenerator.py". The output of this script is a .npz file that is then used in "level_stat_TLI.py".
