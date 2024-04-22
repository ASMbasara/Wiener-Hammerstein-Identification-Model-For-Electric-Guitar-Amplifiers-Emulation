Python implementation of a Wiener-Hammerstein System Identification Model


The flow of how is meant to be running nowadays starts with PWL Signal Generator.py which will generate a PWL signal that is meant to go as an input into a Spice simulator loaded with the circuit of the circuit that is meant to be modelled.

It's supposed to generate two signals: a clean one and a saturated one. The output signals of the circuit must be used as the input of the Filter Designer (FIR filter design of H1 and H2 filters) or the IIR Design - GA (Genetic algorithm design of IIR filters) depending on the type of filter that the user wants to use on the model.

Lastly the Model Optimization script will optimize the model's parameters and together with the designed filters it should generate an approximate digital model of the target circuit .
