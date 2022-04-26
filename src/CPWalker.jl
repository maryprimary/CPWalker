module CPWalker


using LinearAlgebra
using LatticeHamiltonian
using LatticeHamiltonian.SlaterDeterminant

export HamConfig, HSWalker, HamConfig2, HSWalker2
export init_overlap
export clone, copy_to
export step_dtau!, update_overlap!, stablize!
export step_slice!
export popctrl!, weight_rescale!
export CPMeasure, calculate_eqgr!
export postprocess_measurements
export CPSim2, initialize_simulation!, relaxation_simulation
export E_trial_simulation!, initial_meaurements!
export premeas_simulation!, postmeas_simulation
export cal_energy, get_eqgr_without_back


include("hamiltonian.jl")

include("walker.jl")

#include("overlap.jl")

include("popctrl.jl")

include("step.jl")

include("measure.jl")

include("cpsim.jl")


end # module
