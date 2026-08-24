// Compile the production Scheme-A + GMRES imaging path with the direct
// Kirchhoff propagator instead of ButterflyPACK.  Keeping the implementation
// shared makes the benchmark differ only in the 1-D datum -> 2-D block apply.
#define KIRCH_DIRECT_KIRCHHOFF_GMRES 1
#include "frequency_kirchhoff_imaging_bf_gmres.cpp"
