# # Shallow cumulus convection (BOMEX)
#
# This example simulates shallow cumulus convection following the Barbados Oceanographic
# and Meteorological Experiment (BOMEX) intercomparison case [Siebesma2003](@cite).
# BOMEX is a canonical test case for large eddy simulations of shallow cumulus
# convection over a subtropical ocean.

using Breeze
using Oceananigans: Oceananigans
using Oceananigans.Units

using AtmosphericProfilesLibrary
using CairoMakie
using CUDA
using Printf
using Random

Random.seed!(938)
Oceananigans.defaults.FloatType = Float32

arch = GPU()
Nx = Ny = 1024
Nz = 300

#x = y = (0, 6400)
x = y = (0, 6400 * 2)
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

wˢ = Field{Nothing, Nothing, Face}(grid)
wˢ_profile = AtmosphericProfilesLibrary.Bomex_subsidence(FT)
set!(wˢ, z -> wˢ_profile(z))
subsidence = SubsidenceForcing(wˢ)


uᵍ = AtmosphericProfilesLibrary.Bomex_geostrophic_u(FT)
vᵍ = AtmosphericProfilesLibrary.Bomex_geostrophic_v(FT)
geostrophic = geostrophic_forcings(z -> uᵍ(z), z -> vᵍ(z))

# ## Moisture tendency (drying)
#
# A prescribed large-scale drying tendency removes moisture above the cloud layer
# ([Siebesma2003](@citet); Appendix B, Eq. B4). This represents the effects of
# advection by the large-scale circulation.

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

coriolis = FPlane(f=3.76e-5)
microphysics = SaturationAdjustment(equilibrium=WarmPhaseEquilibrium())
momentum_advection = WENO(order=5)
scalar_advection = (; ρθ=WENO(order=5), ρqᵗ = WENO(order=3))
#scalar_advection = WENO(order=3)

model = AtmosphereModel(grid; formulation, coriolis, microphysics, forcing,
                        momentum_advection, scalar_advection,
                        boundary_conditions = (ρθ=ρθ_bcs, ρqᵗ=ρqᵗ_bcs, ρu=ρu_bcs, ρv=ρv_bcs))

FT = eltype(grid)
θˡⁱ₀ = AtmosphericProfilesLibrary.Bomex_θ_liq_ice(FT)
qᵗ₀ = AtmosphericProfilesLibrary.Bomex_q_tot(FT)
u₀ = AtmosphericProfilesLibrary.Bomex_u(FT)

using Breeze.Thermodynamics: dry_air_gas_constant, vapor_gas_constant

Rᵈ = dry_air_gas_constant(constants)
cᵖᵈ = constants.dry_air.heat_capacity
p₀ = reference_state.base_pressure
χ = (p₀ / 1e5)^(Rᵈ/  cᵖᵈ)

δθ = 0.1      # K
δqᵗ = 2.5e-5  # kg/kg
zδ = 1600     # m

ϵ() = rand() - 1/2
θᵢ(x, y, z) = χ * θˡⁱ₀(z) + δθ  * ϵ() * (z < zδ)
qᵢ(x, y, z) = qᵗ₀(z)  + δqᵗ * ϵ() * (z < zδ)
uᵢ(x, y, z) = u₀(z)

set!(model, θ=θᵢ, qᵗ=qᵢ, u=uᵢ)

simulation = Simulation(model; Δt=1, stop_time=2hour)
conjure_time_step_wizard!(simulation, cfl=0.5, IterationInterval(3))

θ = liquid_ice_potential_temperature(model)
qˡ = model.microphysical_fields.qˡ
qᵛ = model.microphysical_fields.qᵛ

u_avg = Field(Average(model.velocities.u, dims=(1, 2)))
v_avg = Field(Average(model.velocities.v, dims=(1, 2)))

wallclock = Ref(time_ns())

function progress(sim)
    compute!(u_avg)
    compute!(v_avg)
    qˡmax = maximum(qˡ)
    qᵗmax = maximum(sim.model.specific_moisture)
    umax = maximum(abs, u_avg)
    vmax = maximum(abs, v_avg)
    elapsed = 1e-9 * (time_ns() - wallclock[])

    msg = @sprintf("Iter: % 5d, t: % 14s, Δt: % 14s, wall time: % 14s, max|ū|: (%.2e, %.2e), max(qᵗ): %.2e, max(qˡ): %.2e",
                   iteration(sim), prettytime(sim), prettytime(sim.Δt), prettytime(elapsed),
                   umax, vmax, qᵗmax, qˡmax)
    @info msg

    wallclock[] = time_ns()

    return nothing
end

add_callback!(simulation, progress, IterationInterval(100))

outputs = merge(model.velocities, model.tracers, (; θ, qˡ, qᵛ))
averaged_outputs = NamedTuple(name => Average(outputs[name], dims=(1, 2)) for name in keys(outputs))

filename_3d = "bomex.jld2"
simulation.output_writers[:fields] = JLD2Writer(model, (; qˡ); filename=filename_3d,
                                                schedule = TimeInterval(30minutes),
                                                overwrite_existing = true)

filename = "bomex_averages.jld2"
simulation.output_writers[:averages] = JLD2Writer(model, averaged_outputs; filename,
                                                  schedule = AveragedTimeInterval(1hour),
                                                  overwrite_existing = true)

# Output horizontal slices at z = 600 m for animation
# Find the k-index closest to z = 600 m
z = Oceananigans.Grids.znodes(grid, Center())
k = searchsortedfirst(z, 800)
@info "Saving slices at z = $(z[k]) m (k = $k)"

u, v, w = model.velocities
slice_fields = (; w, qˡ)
slice_outputs = (
    wxy = view(w, :, :, k),
    qˡxy = view(qˡ, :, :, k),
    wxz = view(w, :, 1, :),
    qˡxz = view(qˡ, :, 1, :),
)

simulation.output_writers[:slices] = JLD2Writer(model, slice_outputs;
                                                filename = "bomex_slices.jld2",
                                                schedule = TimeInterval(30seconds),
                                                overwrite_existing = true)

@info "Running BOMEX simulation..."
run!(simulation)

# ## Animation of horizontal slices
#
# We create an animation showing the evolution of vertical velocity and liquid
# water at z = 800 m, which is near the cloud base level.

wxz_ts = FieldTimeSeries("bomex_slices.jld2", "wxz")
qˡxz_ts = FieldTimeSeries("bomex_slices.jld2", "qˡxz")
wxy_ts = FieldTimeSeries("bomex_slices.jld2", "wxy")
qˡxy_ts = FieldTimeSeries("bomex_slices.jld2", "qˡxy")

times = wxz_ts.times
Nt = length(times)

# Create animation
slices_fig = Figure(size=(800, 1200), fontsize=14)
axwxz = Axis(slices_fig[1, 2], aspect=2, xlabel="x (m)", ylabel="z (m)", title="Vertical velocity w")
axqxz = Axis(slices_fig[1, 3], aspect=2, xlabel="x (m)", ylabel="z (m)", title="Liquid water qˡ")
axwxy = Axis(slices_fig[2, 2], aspect=1, xlabel="x (m)", ylabel="y (m)")
axqxy = Axis(slices_fig[2, 3], aspect=1, xlabel="x (m)", ylabel="y (m)")

# Determine color limits from the data
wlim= maximum(abs, wxz_ts) /  2
qˡlim= maximum(qˡxz_ts) / 2 

n = Observable(1)
wxz_n = @lift wxz_ts[$n]
qˡxz_n = @lift qˡxz_ts[$n]
wxy_n = @lift wxy_ts[$n]
qˡxy_n = @lift qˡxy_ts[$n]
title_text = @lift "BOMEX slices at t = " * prettytime(times[$n])

hmw = heatmap!(axwxz, wxz_n, colormap=:balance, colorrange=(-wlim, wlim))
hmq = heatmap!(axqxz, qˡxz_n, colormap=:dense, colorrange=(0, qˡlim))
hmw = heatmap!(axwxy, wxy_n, colormap=:balance, colorrange=(-wlim, wlim))
hmq = heatmap!(axqxy, qˡxy_n, colormap=:dense, colorrange=(0, qˡlim))

Colorbar(slices_fig[1:2, 1], hmw, label="w (m/s)", flipaxis=false)
Colorbar(slices_fig[1:2, 4], hmq, label="qˡ (kg/kg)")

slices_fig[0, :] = Label(slices_fig, title_text, fontsize=18, tellwidth=false)

# Record animation
CairoMakie.record(slices_fig, "bomex_slices.mp4", 1:Nt, framerate=10) do nn
    n[] = nn
end
