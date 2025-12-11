# TDA for population neural activity

### Topological data anaylsis (TDA) for population neural activity in LGN and V1

- for data collection and preprocessing details: https://doi.org/10.1101/2025.08.25.672226


**Analysis pipeline for:**
1. Training CEBRA* models (GPU/CUDA capabilities recommended)
    - generates neural activity embeddings
2. Running topological data analysis (TDA) using Ripser**
    - constructs a Vietoris-Rips filtration on CEBRA embeddings
3. Comparing topology of embeddings across brain areas, stimuli, & embedding dimensions
    - via persistence diagrams, persistence landscapes, persistence barcode, and betti curves

All steps are in jupyter notebooks with intermediate data files that are saved and reused in subsequent notebooks
- *note:* you can "jump into" a notebook without having generated data from the previous. this will limit the results to the example data


## Data overview

Includes: 
- cleaned neural spike data as primary input (```data/clean_spike_data.zip```)
- example intermediate files 
    - ```data/CEBRA_embedding_examples```
        - limited CERBA embeddings (3d & 8d for LGN & V1 color exchange)
    - ```data/persistence_diagram_examples```
        - limited dgms examples (all dimensions, LGN & V1, color exchange, limited mice)
    - ```data/all_dgms.zip```
        - all persistence diagrams

There are example intermediate data files provided in the ```data/``` folder, but all downstream intermediates can be generated from using ```clean_spike_data.zip``` located in ```data/```

## Enviornment set-up & requirements
If using a machine with CUDA capabilities: ```conda env create -f tda_environment.yml``` & ```conda activate topology```

If using CPU: ```conda env create -f tda_environment_cpu.yml``` & ```conda activate base```

<br> To check GPU availability: ```import torch``` & ```torch.cuda.is_available()``` 

RAM: recommended at least 8GB

## Running the pipeline

All numerical steps in the notebooks should be executed in order. <br> If there are multiple processing options per step, choose the desired method and run the cell before moving on to the next.

1. ***01_CEBRA_embeddings.ipynb***
   - load cleaned spike data tensors from ```\data\``` folder
   - train CEBRA models
   - save embeddings to pickle files
   - visualize embeddings
     
2. ***02_Ripser.ipynb***
   - load embedding pickle files
   - compute persistence homology using Ripser
       - generate dgms (persistence diagrams) for each homology group (H0, H1, & H2)
   - plot dgms on a persistence diagram
   - save dgms to pickle files

3. ***03_betti_curves.ipynb***
   - load dgms pickle files
   - generate betti curves
         - plot betti numbers for each homology group during filtration

4. ***04_persistence_landscapes.ipynb***
    - load dgms pickle files
    - compute average persistence landscapes across embeddings, regions, & stimuli
         - persistence landscape is a vectorized representation of a persistence diagram. these plots are assumed to be more stable than betti curves and displays topological features as functions. often used as input to downstream pipelines like machine learning predictive models or statistical analyses.
           
<br>

Other notebook:
***Persistence_barcodes.ipynb*** 

  - fuzzy UMAP algorithm from Gardner et al*** to generate persistence barcodes
  - does not generate embeddings but rather constructs a UMAP neighborhood graph for input into Ripser

<br>
* https://cebra.ai/docs/

<br>

** https://ripser.scikit-tda.org/en/latest/index.html

***Gardner, R.J., Hermansen, E., Pachitariu, M. et al. Toroidal topology of population activity in grid cells. Nature 602, 123–128 (2022). https://doi.org/10.1038/s41586-021-04268-7
