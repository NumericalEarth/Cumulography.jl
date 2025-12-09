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

@inline w_dz_ϕ(i, j, k, grid, w, ϕ) = @inbounds w[i, j, k] * ∂zᶜᶜᶠ(i, j, k, grid, ϕ)

@inline function Fρu_subsidence(i, j, k, grid, clock, fields, p)
    w_dz_U = ℑzᵃᵃᶜ(i, j, k, grid, w_dz_ϕ, p.wˢ, p.u_avg)
    return @inbounds - p.ρᵣ[i, j, k] * w_dz_U
end

@inline function Fρv_subsidence(i, j, k, grid, clock, fields, p)
    w_dz_V = ℑzᵃᵃᶜ(i, j, k, grid, w_dz_ϕ, p.wˢ, p.v_avg)
    return @inbounds - p.ρᵣ[i, j, k] * w_dz_V
end

wˢ = AtmosphericProfilesLibrary.Bomex_subsidence(FT)
subsidence = SubsidenceForcing(wˢ)

coriolis = FPlane(f=3.76e-5)

uᵍ = AtmosphericProfilesLibrary.Bomex_geostrophic_u(FT)
vᵍ = AtmosphericProfilesLibrary.Bomex_geostrophic_v(FT)
geostrophic = geostrophic_forcings(uᵍ, vᵍ)

ρᵣ = formulation.reference_state.density
drying = Field{Nothing, Nothing, Center}(grid)
dqdt_profile = AtmosphericProfilesLibrary.Bomex_dqtdt(FT)
set!(drying, z -> dqdt_profile(z))
set!(drying, ρᵣ * drying)
ρqᵗ_drying_forcing = Forcing(drying)

Fρe_field = Field{Nothing, Nothing, Center}(grid)
cᵖᵈ = constants.dry_air.heat_capacity
dTdt_bomex = AtmosphericProfilesLibrary.Bomex_dTdt(FT)
set!(Fρe_field, z -> dTdt_bomex(1, z))
set!(Fρe_field, ρᵣ * cᵖᵈ * Fρe_field)
ρe_radiation_forcing = Forcing(Fρe_field)

ρu_forcing = (subsidence, geostrophic.ρu)
ρv_forcing = (subsidence, geostrophic.ρv)
ρqᵗ_forcing = (ρqᵗ_drying_forcing, subsidence)
ρθ_forcing = subsidence
ρe_forcing = ρe_radiation_forcing

forcing = (; ρu=ρu_forcing, ρv=ρv_forcing, ρθ=ρθ_forcing,
             ρe=ρe_forcing, ρqᵗ=ρqᵗ_forcing)
nothing #hide

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
