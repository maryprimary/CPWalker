module CPWalker


using LinearAlgebra
using LatticeHamiltonian
using LatticeHamiltonian.SlaterDeterminant

export HamConfig, HSWalker, HamConfig2, HSWalker2, HamConfig3, HSWalker3
export init_overlap
export clone, copy_to
export step_dtau!, update_overlap!, stablize!, noorthstablize!, decorate_stablize!
export step_slice!
export popctrl!, weight_rescale!
export CPMeasure, calculate_eqgr!
export postprocess_measurements
export CPSim2, initialize_simulation!, relaxation_simulation
export CPSim3, update_backwalkers!
export E_trial_simulation!, initial_meaurements!
export premeas_simulation!, postmeas_simulation
export cal_energy, get_eqgr_without_back
export save_density_profile


include("hamiltonian.jl")

include("hmltnh.jl")

include("walker.jl")

include("walkernh.jl")

#include("overlap.jl")

include("popctrl.jl")

include("step.jl")

include("stepnh.jl")

include("measure.jl")

include("cpsim.jl")

include("cpsimnh.jl")


end # module
