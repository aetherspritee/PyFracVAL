# TunableCCA
Implementation of FracVAL by [Morán, J. et al. 2019](https://www.sciencedirect.com/science/article/pii/S0010465519300323?via%3Dihub) in python.

## TODO
 - [ ] Fix PCA issues 
   + [X] Instability with arcos arguments, tied to monomer size somehow
   + [ ] low fractal dimensions fail -> This is also happening for original FracVAL
   + [ ] high fractal prefactors fail -> This is also happening for original FracVAL
   + [X] yet to be understood infinite loop occurs sometimes
 - [ ] Allow distribution functions for monomer radii
 - [ ] Parallelize
 - [ ] High fractal dimensions are super slow -> Happens for FracVAL aswell, bit more noticeable for this implementation i feel
