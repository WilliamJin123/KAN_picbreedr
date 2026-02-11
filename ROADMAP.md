### Phases

- 1: Figure out how to get the CPPNs, draw the heat maps of each layer, from the fer repository
    - Use the skull, apple, butterfly (earth/candle genomes not public — email akarshkumar0101@gmail.com to request)
    - **CPPNs:** Run `references/fer/src/process_pb.py --zip_path=<genome.zip> --save_dir=<out>` to layerize NEAT genomes into dense MLPs. Precomputed data already in `references/fer/data/`.
    - **Heat maps:** Call `cppn.generate_image(params, return_features=True)` → returns per-layer activations as (H,W,neurons) tensors, plot each neuron with `imshow(cmap='bwr_r', vmin=-1, vmax=1)`. See `fer.ipynb` cell 16-17.
    - **Weight sweeps:** Params are flattened to a 1D vector via `FlattenCPPNParameters`. Use `sweep_weight(params, weight_id, cppn, r=1, n=5)` from `fer.ipynb` cell 21 to vary a single weight ±1 and generate image strips. Interesting weight IDs found by sweeping 200 random directions and sorting by variance (cell 29-31). Picbreeder weights control clean semantic features (mouth, eyes, wings); SGD weights produce entangled chaos.
- 2: Train basic KAN on all the images
    - Same thing
- 3: Train swarm-based KAN on all the images
    - Same thing
- 4: Train memetic based KAN on all the images with variable weights
    - Same thing
- 4.1: See if the memetic based KAN can reverse engineer original images (when we reset all weights to 1) from the mutated images (ex. wide skull, apple leaf angle, etc.)
- 5: Memetic KAN on textual datasets
    - See if we can get interesting generalization properties / recovering a "purer" form of data
    