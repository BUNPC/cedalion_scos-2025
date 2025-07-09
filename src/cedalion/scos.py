import numpy as np
import pickle
import xarray as xr
import os

def test():
    print('This worked!\nCongradulations!\nYou have successfully loaded the SCOS fork.\nYou now have access to scos.py')
    return

def compute_autocorrelation_mcx(
    fwm,
    date_time_str,
    aDb = np.array([1e-6,1e-6,6e-6,1e-6,1e-6,6e-6,]),             # should be mm^2/s   weighted diffusion brain
    wavelength = 852e-6,                                          # mm (hence e-6)
    n = 1.33,                                                     # refractive index
    T_exp = 4000e-6,                                              # camera exposure time in seconds
    NUM_TAU = 128,                                                # number of Decorrelation times to simulate
    data_dir = os.getcwd()                                        # directory where the data will save
):
    """Calculates the temporal field autocorrelation for a given head model and probe geometry. 
    
    Args:
        fwm: Forward model object based on the the given head model and probe geometry from the .snirf file. This should be generated using cedalion.imagereco.forward_model.ForwardModel(head, ninja_snapped_aligned, meas_list)
        date_time_str: String containing the date and time the code is run. This is used for file naming.
        aDb: Numpy array containing values for the weighted diffusion coefficient of the brain (mm^2/s) defined for each tissue type. Values from (lin2023).
        wavelength: Wavelength of light in mm.
        n: refractive index used for the head model.
        T_exp: Exposure time of the cameras in seconds.
        NUM_TAU: The number of non-zero decorrelation times used for the simulation.
        data_dir: directory where the data will save
        
    Returns: 
        Returns none, but saves two variables:
            {date_time_str}data_DCS_optode.pickle: A dictionary containing the temporal field autocorrelation for each channel (source-detector pair) in the probe geometry. 
            {date_time_str}data_DCS_all_chunk.pickle: A dictionary containing the temporal field autocorrelation for each optode in the probe geometry to each voxel in the head model.
     
    ### note! tissue properties is 6 tissue types each with 4 optical properties
    # 0 = absorption 
    # 1 = scattering
    # 2 = anisotropy
    # 3 = refraction
    # tissue layers are 
    # 0 = skin
    # 1 = skull
    # 2 = dm
    # 3 = csf
    # 4 = gm
    # 5 = wm
    """

    data_DCS_all = {}
    data_DCS_optode = {}

    # save absorption 
    mua = fwm.tissue_properties[:,0]           # 1/mm   abs coeff
    mus = fwm.tissue_properties[:,1]           # 1/mm   scatter coeff

    # define k0
    k0 = 2*np.pi/wavelength*np.ones(len(mua))  # 1/mm          angular wavenumber

    # Change refractive index
    fwm.tissue_properties[:,3] = n*np.ones(len(mua))   # change refractive index

    # Define the range of decorrelation times for the MCX simulation
    tau_DCS = np.array([0])
    tau_DCS = np.append(tau_DCS,np.logspace(-7, np.log10(T_exp), NUM_TAU))  # define tau range in seconds
    
    # Itterate for each decorrelation time
    for tau in range(len(tau_DCS)):       #tau in seconds
        # Change the absorption to be decorrelation time dependent
        fwm.tissue_properties[:,0] = mua + 2 * aDb * k0**2 * (mus + mua) * tau_DCS[tau]

        # Run MCX
        fluence_all, fluence_at_optodes = fwm.compute_fluence_mcx()
        data_DCS_all[f'data_all_tau{tau}'] = fluence_all
        data_DCS_optode[f'data_optode_tau{tau}'] = fluence_at_optodes

    # Save the optode data    
    with open(f'{data_dir}\\{date_time_str}data_DCS_optode.pickle', 'wb') as f:
        pickle.dump(data_DCS_optode, f)

    # Save the voxel data in chunks
    with open(f'{data_dir}\\{date_time_str}data_DCS_all_chunk.pickle', 'wb') as f:
        for key, value in data_DCS_all.items():
            pickle.dump((key, value), f)
            print(key)
    
def compute_sensitivity_scos(
    fwm,
    rec,
    date_time_str,
    aDb = np.array([1e-6,1e-6,6e-6,1e-6,1e-6,6e-6,]),             # should be mm^2/s   weighted diffusion brain
    wavelength = 852e-6,                                          # mm (hence e-6)
    n = 1.33,                                                     # refractive index
    T_exp = 4000e-6,                                              # camera exposure time in seconds
    NUM_TAU = 128,                                                # number of Decorrelation times to simulate
    dDb = 1e-7,                                                   # brain diffusion perturbation
    S = 1,                                                        # Source term
    v = 1,                                                        # Speed of light
    beta = 1,                                                     # Source coherence parameter
    data_dir = os.getcwd()                                        # directory where the data will save
):
    """Calculate the sensitivity matrix for SCOS given the MCX output for the baseline temporal field autocorrelation using the rytov approximation.
    
    Args:
        fwm: Forward model object based on the the given head model and probe geometry from the .snirf file. This should be generated using cedalion.imagereco.forward_model.ForwardModel(head, ninja_snapped_aligned, meas_list)
        rec: Recording snirf object.
        date_time_str: String containing the date and time the code is run. This is used for file naming.
        aDb: Numpy array containing values for the weighted diffusion coefficient of the brain (mm^2/s) defined for each tissue type. Values from (lin2023).
        wavelength: Wavelength of light in mm.
        n: refractive index used for the head model.
        T_exp: Exposure time of the cameras.
        NUM_TAU: The number of non-zero decorrelation times used for the simulation.
        dDb: The perterbation in the diffusion coefficient of the brain.
        S: Source term. Changing this does not impact results showing relative activation.
        v: Speed of light. Changing this does not impact results showing relative activation.
        beta: Source coherence parameter. 
        data_dir: directory where the data will save

    Returns: 
        xr.DataArray: Sensitivity matrix for each channel and vertex. Also saves the sensitivity matrix.
     
    ### note! tissue properties is 6 tissue types each with 4 optical properties
    # 0 = absorption 
    # 1 = scattering
    # 2 = anisotropy
    # 3 = refraction
    # tissue layers are 
    # 0 = skin
    # 1 = skull
    # 2 = dm
    # 3 = csf
    # 4 = gm
    # 5 = wm
    """
    # Get the measurement list from the foward model object
    meas_list = rec._measurement_lists['hrf_conc']

    T_exp = np.float64(T_exp)

    mua = fwm.tissue_properties[:,0]           # 1/mm   abs coeff
    mus = fwm.tissue_properties[:,1]           # 1/mm   scatter coeff
    k0 = 2*np.pi/wavelength*np.ones(len(mua))  # 1/mm          angular wavenumber
    mus_reduced = mus[5]*(1-fwm.tissue_properties[5,2])      # for brain, but it cancels out anyway    from us' = us(1-g)

    # Define the photon diffusion coefficient
    Dp = v/(3*mus_reduced)

    # Change refractive index
    fwm.tissue_properties[:,3] = n*np.ones(len(mua))   

    # Define the range of decorrelation times
    tau_DCS = np.array([0])
    tau_DCS = np.append(tau_DCS,np.logspace(-7, np.log10(T_exp), NUM_TAU))  # define tau range in seconds

    # This supports future implementation if the last value of Tau does not equal T_exp
    tau_to_integrate = []
    for tau in range(NUM_TAU+1):
        if tau_DCS[tau]<=T_exp*1.00001:          # multiply to make T_exp slightly bigger to include last point that is excluded due to rounding error (in other words the last value in tau_DCS is slightly higher than tau exp due to a round up when taking the exponent of a log)
            tau_to_integrate.append(tau_DCS[tau])

    # Define the differential volume element for the head model
    d3 = fwm.unitinmm**3  

    # Load the data_DCS_optode pickle file
    with open(f'{data_dir}\\{date_time_str}data_DCS_optode.pickle', 'rb') as f:
        data_DCS_optode = pickle.load(f)

    # Load the data_DCS_all  chuncked pickle file
    data_DCS_all = {}
    with open(f'{data_dir}\\{date_time_str}data_DCS_all_chunk.pickle', 'rb') as f:
        while True:
            try:
                key, value = pickle.load(f)
                print(key)
                # print(value)
                data_DCS_all[key] = value
            except EOFError:
                break

    # Stack the data for matrix multiplication
    for tau in range(len(tau_to_integrate)):
        key = f'data_all_tau{int(tau)}'
        if data_DCS_all[key].ndim > 3:
            data_DCS_all[key] = data_DCS_all[key].stack(voxel = ['i','j','k'])

    # Define the number of elements needed for various preallocation
    num_voxels = np.max(np.shape(data_DCS_all[key]))
    num_channels = len(meas_list.source)/2
    num_channels = int(num_channels)
    num_voxels = int(num_voxels)
    num_vertices = fwm.head_model.voxel_to_vertex_brain.shape[1]

    # Preallocate variables
    G10sd = np.zeros(len(tau_to_integrate))             # Baseline temporal field autocorrelation from source to detector
    phi   = G10sd                                       # Perturbation term
    G1sd  = G10sd                                       # Perturbed temporal field autocorrelation from source to detector
    A = np.zeros([num_channels, num_vertices])          # Sensitivity matrix
    ALLs = np.zeros((num_voxels,len(tau_to_integrate))) # Baseline temporal field autocorrelation from source to voxel 
    ALLd = np.zeros((num_voxels,len(tau_to_integrate))) # Baseline temporal field autocorelation from detector to voxel
    OPTODE = np.zeros((len(tau_to_integrate)))          # Baseline temporal field autocorrelation from source to detector

    # Define the constant term for calculating the perturbation
    C = np.float64(d3*2*mus_reduced*k0[0]**2*dDb/S)

    # Run the MCX once for each decorrelation time
    for i in range(num_channels):  # channel
        print(i)
        # get channel
        source = meas_list.source[i]
        detector = meas_list.detector[i]
        # pull data from channel to an np array with all tau and voxels
        for tau in range(len(tau_to_integrate)):  # tau time shift
            keyO = f'data_optode_tau{int(tau)}'
            keyA = f'data_all_tau{int(tau)}'
            ALLs[:,tau] = data_DCS_all[keyA].sel(label = source, wavelength = 'nan').values     # source corr fluence   v byt tau
            ALLd[:,tau] = data_DCS_all[keyA].sel(label = detector, wavelength = 'nan').values   # detector corr fluence  v by tau
            OPTODE[tau] = data_DCS_optode[keyO].sel(optode1 = source, optode2 = detector, wavelength = 'nan').values # channel corr fluence 1d tau

        # Calculate the perturbation term
        phi = (-(np.divide(C*(np.matlib.repmat(tau_to_integrate,num_voxels,1)*ALLd*ALLs ),np.matlib.repmat(OPTODE,num_voxels,1))).T @ fwm.head_model.voxel_to_vertex_brain).T

        # set nan and inf to 0
        phi[np.isnan(phi) | np.isinf(phi)] = 0
        
        # Calculate the perturbed temporal field autocorrelation of each channel using the Rytov Approximation
        G1sd = np.matlib.repmat(OPTODE,num_vertices,1)*np.exp(phi)

        # Make the baseline temporal field autocorrelation of each channel the correct size for matrix opperations
        G10sd = np.matlib.repmat(OPTODE,num_vertices,1)

        # Integrate both autocorrelations to get contrast and solve for the sensitivity matrix
        denominator =    np.trapz(  np.abs(np.divide(G1sd ,np.transpose(np.matlib.repmat(G1sd[:,0] ,len(tau_to_integrate),1))))**2*np.matlib.repmat((1-tau_to_integrate/T_exp),num_vertices,1)  , np.matlib.repmat(tau_to_integrate,num_vertices,1), axis=1)  # perterbed 
        numerator =      np.trapz(  np.abs(np.divide(G10sd,np.transpose(np.matlib.repmat(G10sd[:,0],len(tau_to_integrate),1))))**2*np.matlib.repmat((1-tau_to_integrate/T_exp),num_vertices,1)  , np.matlib.repmat(tau_to_integrate,num_vertices,1), axis=1)  # original 
        A[i,:] = (np.divide(numerator,denominator)-1)/dDb

    # Put A into an xarray
    A_xr = xr.DataArray(
        A[::3,:,np.newaxis],
        dims=["channel", "vertex", 'wavelength'],
        coords={
            "channel": ("channel", rec['hrf_conc'].channel.values),
            "wavelength": ("wavelength", [830]),
            "is_brain": ("vertex", np.ones(num_vertices)),
        })

    with open(f'{data_dir}\\{date_time_str}_sensitivity_matrix_rytov_xr.pickle', 'wb') as f:
        pickle.dump(A_xr, f)

    return A_xr