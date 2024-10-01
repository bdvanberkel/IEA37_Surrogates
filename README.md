# IEA37_Surrogates
Repository for the IEA37 3.4MW and 10MW turbine-level load, power and ct surrogates I used together with PyWake to obtain a dataset to train the Graph Neural Network wind farm simulation surrogate I used in WakeWISE. The 15MW turbine I did not end up making a surrogate model for, as there is a strong edgewise instability in OpenFAST that prevented me from getting a good dataset for training. I had to do some cleaning of the 10MW dataset to remove runs with the edgewise instability. The 3.4MW turbine did not suffer from this.

## More info
I will add more info about the generation of the dataset, and the files used to generate the dataset, later on. Quick info for now: 2^15 datapoints generated for the dataset, with random wind speeds, yaw angles, turbulence intensities and shear exponents. Turbulence intensity according to IEC61400-1 standard. Wind speed uniformly sampled in the range [0-30]. More info will be added later.
