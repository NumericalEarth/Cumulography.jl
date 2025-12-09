# This script setus up a simulation of shallow cumulus convection following the BOMEX case
# described by Siebesma et al. (2003).
#
# The "clouds" are represneted by the liquid water concentration, denoted qˡ.
#
# With WarmPhaseSaturationAdjustment, AtmosphereModel stores qˡ in
#
# model.microphysical_fields.qˡ
# 
# It is a three dimensional field.

using Breeze
using Oceananigans
using Oceananigans.Units

using AtmosphericProfilesLibrary
using Printf
using CairoMakie

#####
##### Grid and model setup
#####

Oceananigans.defaults.FloatType = Float32 # will speed things up esp on GPU
arch = CPU()

Nx = Ny = 32
Nz = 75

x = y = (0, 6400)
z = (0, 3000)

using Oceananigans.Operators: ∂zᶜᶜᶠ, ℑzᵃᵃᶜ

grid = RectilinearGrid(arch; x, y, z,
                       size = (Nx, Ny, Nz), halo = (5, 5, 5),
                       topology = (Periodic, Periodic, Bounded))

constants = ThermodynamicConstants()

reference_state = ReferenceState(grid, constants,
                                 base_pressure = 101500,
                                 potential_temperature = 299.1)

formulation = AnelasticFormulation(reference_state,
                                   thermodynamics = :LiquidIcePotentialTemperature)

w′θ′ = 8e-3     # K m/s (sensible heat flux)
w′qᵗ′ = 5.2e-5  # m/s (moisture flux)

FT = eltype(grid)
p₀ = reference_state.base_pressure
θ₀ = reference_state.potential_temperature
q₀ = Breeze.Thermodynamics.MoistureMassFractions{FT} |> zero
ρ₀ = Breeze.Thermodynamics.density(p₀, θ₀, q₀, constants)

ρθ_bcs = FieldBoundaryConditions(bottom=FluxBoundaryCondition(ρ₀ * w′θ′))
ρqᵗ_bcs = FieldBoundaryConditions(bottom=FluxBoundaryCondition(ρ₀ * w′qᵗ′))

u★ = 0.28  # m/s
@inline ρu_drag(x, y, t, ρu, ρv, p) = - p.ρ₀ * p.u★^2 * ρu / sqrt(ρu^2 + ρv^2)
@inline ρv_drag(x, y, t, ρu, ρv, p) = - p.ρ₀ * p.u★^2 * ρv / sqrt(ρu^2 + ρv^2)

ρu_drag_bc = FluxBoundaryCondition(ρu_drag, field_dependencies=(:ρu, :ρv), parameters=(; ρ₀, u★))
ρv_drag_bc = FluxBoundaryCondition(ρv_drag, field_dependencies=(:ρu, :ρv), parameters=(; ρ₀, u★))
ρu_bcs = FieldBoundaryConditions(bottom=ρu_drag_bc)
ρv_bcs = FieldBoundaryConditions(bottom=ρv_drag_bc)

# ## Large-scale subsidence
#
# The BOMEX case includes large-scale subsidence that advects mean profiles downward.
# The subsidence velocity profile is prescribed by [Siebesma2003](@citet); Appendix B, Eq. B5:
# ```math
# w^s(z) = \begin{cases}
#   W^s \frac{z}{z_1} & z \le z_1 \\
#   W^s \left ( 1 - \frac{z - z_1}{z_2 - z_1} \right ) & z_1 < z \le z_2 \\
#   0 & z > z_2
# \end{cases}
# ```
# where ``W^s = -6.5 \times 10^{-3}`` m/s (note the negative sign for "subisdence"),
# ``z_1 = 1500`` m and ``z_2 = 2100`` m.
#
# The subsidence velocity profile is provided by [AtmosphericProfilesLibrary](https://github.com/CliMA/AtmosphericProfilesLibrary.jl),

wˢ = Field{Nothing, Nothing, Face}(grid)
wˢ_profile = AtmosphericProfilesLibrary.Bomex_subsidence(FT)
set!(wˢ, z -> wˢ_profile(z))

# and looks like:

lines(wˢ; axis = (xlabel = "wˢ (m/s)",))

# We apply subsidence as a forcing term to the horizontally-averaged prognostic variables.
# This requires computing horizontal averages at each time step and storing them in
# fields that can be accessed by the forcing functions.

@inline w_dz_ϕ(i, j, k, grid, w, ϕ) = @inbounds w[i, j, k] * ∂zᶜᶜᶠ(i, j, k, grid, ϕ)

@inline function Fρu_subsidence(i, j, k, grid, clock, fields, p)
    w_dz_U = ℑzᵃᵃᶜ(i, j, k, grid, w_dz_ϕ, p.wˢ, p.u_avg)
    return @inbounds - p.ρᵣ[i, j, k] * w_dz_U
end

@inline function Fρv_subsidence(i, j, k, grid, clock, fields, p)
    w_dz_V = ℑzᵃᵃᶜ(i, j, k, grid, w_dz_ϕ, p.wˢ, p.v_avg)
    return @inbounds - p.ρᵣ[i, j, k] * w_dz_V
end

@inline function Fρθ_subsidence(i, j, k, grid, clock, fields, p)
    w_dz_Θ = ℑzᵃᵃᶜ(i, j, k, grid, w_dz_ϕ, p.wˢ, p.θ_avg)
    return @inbounds - p.ρᵣ[i, j, k] * w_dz_Θ
end

@inline function Fρqᵗ_subsidence(i, j, k, grid, clock, fields, p)
    w_dz_Qᵗ = ℑzᵃᵃᶜ(i, j, k, grid, w_dz_ϕ, p.wˢ, p.qᵗ_avg)
    return @inbounds - p.ρᵣ[i, j, k] * w_dz_Qᵗ
end

# Next, we build horizontally-averaged fields for subsidence. We suffix these `_f` for "forcing".
# After we construct the model and simulation, we will write a callback that computes these
# horizontal averages every time step.

u_avg = Field{Nothing, Nothing, Center}(grid)
v_avg = Field{Nothing, Nothing, Center}(grid)
θ_avg = Field{Nothing, Nothing, Center}(grid)
qᵗ_avg = Field{Nothing, Nothing, Center}(grid)

ρᵣ = formulation.reference_state.density
ρu_subsidence_forcing = Forcing(Fρu_subsidence, discrete_form=true, parameters=(; u_avg, wˢ, ρᵣ))
ρv_subsidence_forcing = Forcing(Fρv_subsidence, discrete_form=true, parameters=(; v_avg, wˢ, ρᵣ))
ρθ_subsidence_forcing = Forcing(Fρθ_subsidence, discrete_form=true, parameters=(; θ_avg, wˢ, ρᵣ))
ρqᵗ_subsidence_forcing = Forcing(Fρqᵗ_subsidence, discrete_form=true, parameters=(; qᵗ_avg, wˢ, ρᵣ))

# ## Geostrophic forcing
#
# The momentum equations include a Coriolis force with prescribed geostrophic wind.
# The geostrophic wind profiles are given by [Siebesma2003](@citet); Appendix B, Eq. B6.

coriolis = FPlane(f=3.76e-5)

uᵍ = Field{Nothing, Nothing, Center}(grid)
vᵍ = Field{Nothing, Nothing, Center}(grid)
uᵍ_profile = AtmosphericProfilesLibrary.Bomex_geostrophic_u(FT)
vᵍ_profile = AtmosphericProfilesLibrary.Bomex_geostrophic_v(FT)
set!(uᵍ, z -> uᵍ_profile(z))
set!(vᵍ, z -> vᵍ_profile(z))
ρuᵍ = Field(ρᵣ * uᵍ)
ρvᵍ = Field(ρᵣ * vᵍ)

@inline Fρu_geostrophic(i, j, k, grid, clock, fields, p) = @inbounds - p.f * p.ρvᵍ[i, j, k]
@inline Fρv_geostrophic(i, j, k, grid, clock, fields, p) = @inbounds + p.f * p.ρuᵍ[i, j, k]

ρu_geostrophic_forcing = Forcing(Fρu_geostrophic, discrete_form=true, parameters=(; f=coriolis.f, ρvᵍ))
ρv_geostrophic_forcing = Forcing(Fρv_geostrophic, discrete_form=true, parameters=(; f=coriolis.f, ρuᵍ))

# ## Moisture tendency (drying)
#
# A prescribed large-scale drying tendency removes moisture above the cloud layer
# ([Siebesma2003](@citet); Appendix B, Eq. B4). This represents the effects of
# advection by the large-scale circulation.

drying = Field{Nothing, Nothing, Center}(grid)
dqdt_profile = AtmosphericProfilesLibrary.Bomex_dqtdt(FT)
set!(drying, z -> dqdt_profile(z))
set!(drying, ρᵣ * drying)
ρqᵗ_drying_forcing = Forcing(drying)

# ## Radiative cooling
#
# A prescribed radiative cooling profile is applied to the thermodynamic equation
# ([Siebesma2003](@citet); Appendix B, Eq. B3). Below the inversion, radiative cooling
# of about 2 K/day counteracts the surface heating.

Fρe_field = Field{Nothing, Nothing, Center}(grid)
cᵖᵈ = constants.dry_air.heat_capacity
dTdt_bomex = AtmosphericProfilesLibrary.Bomex_dTdt(FT)
set!(Fρe_field, z -> dTdt_bomex(1, z))
set!(Fρe_field, ρᵣ * cᵖᵈ * Fρe_field)
ρe_radiation_forcing = Forcing(Fρe_field)

# ## Assembling all the forcings
#
# We build tuples of forcings for all the variables. Note that forcing functions
# are provided for both `ρθ` and `ρe`, which both contribute to the tendency of `ρθ`
# in different ways. In particular, the tendency for `ρθ` is written
#
# ```math
# ∂_t (ρ θ) = - ∇ ⋅ ( ρ \boldsymbol{u} θ ) + F_{ρθ} + \frac{1}{cᵖᵐ Π} F_{ρ e} + \cdots
# ```
#
# where ``F_{ρ e}`` denotes the forcing function provided for `ρe` (e.g. for "energy density"),
# ``F_{ρθ}`` denotes the forcing function provided for `ρθ`, and the ``\cdots`` denote
# additional terms.

ρu_forcing = (ρu_subsidence_forcing, ρu_geostrophic_forcing)
ρv_forcing = (ρv_subsidence_forcing, ρv_geostrophic_forcing)
ρqᵗ_forcing = (ρqᵗ_drying_forcing, ρqᵗ_subsidence_forcing)
ρθ_forcing = ρθ_subsidence_forcing
ρe_forcing = ρe_radiation_forcing

forcing = (; ρu=ρu_forcing, ρv=ρv_forcing, ρθ=ρθ_forcing,
             ρe=ρe_forcing, ρqᵗ=ρqᵗ_forcing)

microphysics = SaturationAdjustment(equilibrium=WarmPhaseEquilibrium())
advection = WENO(order=9)

model = AtmosphereModel(grid; formulation, coriolis, microphysics, advection, forcing,
                        boundary_conditions = (ρθ=ρθ_bcs, ρqᵗ=ρqᵗ_bcs, ρu=ρu_bcs, ρv=ρv_bcs))

#####
##### Initial conditions
#####

FT = eltype(grid)
θˡⁱ₀ = AtmosphericProfilesLibrary.Bomex_θ_liq_ice(FT)
qᵗ₀ = AtmosphericProfilesLibrary.Bomex_q_tot(FT)
u₀ = AtmosphericProfilesLibrary.Bomex_u(FT)

using Breeze.Thermodynamics: dry_air_gas_constant, vapor_gas_constant

Rᵈ = dry_air_gas_constant(constants)
cᵖᵈ = constants.dry_air.heat_capacity
p₀ = reference_state.base_pressure
χ = (p₀ / 1e5)^(Rᵈ/  cᵖᵈ)

δθ = 0.1     # K
δqᵗ = 2.5e-5 # kg/kg
zδ = 1600    # m

ϵ() = rand() - 1/2
θᵢ(x, y, z) = χ * θˡⁱ₀(z) + δθ  * ϵ() * (z < zδ)
qᵢ(x, y, z) = qᵗ₀(z)  + δqᵗ * ϵ() * (z < zδ)
uᵢ(x, y, z) = u₀(z)

set!(model, θ=θᵢ, qᵗ=qᵢ, u=uᵢ)

##### 
##### Simulation
##### 

simulation = Simulation(model; Δt=10, stop_time=1hour)
conjure_time_step_wizard!(simulation, cfl=0.7)

θ = liquid_ice_potential_temperature(model)
qˡ = model.microphysical_fields.qˡ
qᵛ = model.microphysical_fields.qᵛ

u_avg = Field(Average(model.velocities.u, dims=(1, 2)))
v_avg = Field(Average(model.velocities.v, dims=(1, 2)))

function progress(sim)
    compute!(u_avg)
    compute!(v_avg)
    qˡmax = maximum(qˡ)
    qᵗmax = maximum(sim.model.specific_moisture)
    umax = maximum(abs, u_avg)
    vmax = maximum(abs, v_avg)

    msg = @sprintf("Iter: %d, t: %s, Δt: %s, max|ū|: (%.2e, %.2e), max(qᵗ): %.2e, max(qˡ): %.2e",
                   iteration(sim), prettytime(sim), prettytime(sim.Δt),
                   umax, vmax, qᵗmax, qˡmax)
    @info msg
    return nothing
end

add_callback!(simulation, progress, IterationInterval(100))

##### 
##### Output
##### 

outputs = merge(model.velocities, (; θ, qˡ, qᵛ))

simulation.output_writers[:cloud] = JLD2Writer(model, outputs;
                                               filename = "bomex.jld2",
                                               schedule = TimeInterval(30minutes),
                                               overwrite_existing = true)

@info "Running BOMEX simulation..."
run!(simulation)
