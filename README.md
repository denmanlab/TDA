# TDA for population neural activity

Pipeline for:
1. Training CEBRA models (GPU/CUDA required)
2. Running topological data analysis (TDA) using Ripser
3. Comparing topology across brain regions, stimuli, and embeddings
    - persistence diagrams, persistence landscapes, persistence barcode, and betti curves

All steps are in jupyter notebooks with intermediate data files saved and reused in later notebooks

<br>

There are example intermediate data files provided in the ```data/``` folder, but all downstream intermediates can be generated from using ```clean_spike_data.zip``` located in ```data/```

# Enviornment set-up
use ```conda env create -f tda_environment.yml```

**Note:** this pipeline uses CUDA-enabled NVIDIA GPU to train and handle CEBRA models.
<br> To check GPU availability: ```import torch``` & ```torch.cuda.is_available()``` -> if ```False``` you may not be able to handle CEBRA models. There may be a CPU work-around, but that has not been tested with this pipeline.

# Running the pipeline

1. 01_CEBRA_embeddings.ipynb
   - loads cleaned spike data tensors from \data\ folder
   - trains CEBRA models
   - saves embeddings to pickle files
   - visualize embeddings
     
2. 02_Ripser.ipynb
   - loads embedding pickle files
   - computes persistence homology using Ripser
       - generates dgms - or persistence diagrams for each homology group (H0, H1, & H2)
   - plot dgms on a persistence diagram
   - saves dgms to pickle files

3. 03_betti_curves.ipynb
   - loads dgms pickle files
   - generates betti curves
         - plots betti numbers for each homology group during filtration

4. 04_persistence_landscapes.ipynb
    - loads dgms pickle files
    - computes average persistence landscapes across embeddings, regions, & stimuli
         - persistence landscape is a vectorized representation of a persistence diagram. these plots are assumed to be more stable than betti curves and displays topological features as functions. often used as input to downstream pipelines like machine learning predictive models or statistical analyses.
           
<br> <br>
Other:
Persistence_barcodes.ipynb
  - uses fuzzy UMAP algorithm from Gardner et al* to generate persistence barcodes
  - does not generate embeddings but rather constructs a UMAP neighborhood graph for input into Ripser


*Gardner, R.J., Hermansen, E., Pachitariu, M. et al. Toroidal topology of population activity in grid cells. Nature 602, 123–128 (2022). https://doi.org/10.1038/s41586-021-04268-7
