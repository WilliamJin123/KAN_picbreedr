# KAN_picbreeder
Training a Kolmogorov-Arnold Network on PicBreedr images. Experimenting with memetic algorithms to jointly evolve weights and activation functions for network interpretability

## Inspo

- [Towards a Platonic Intelligence with Unified Factored Representations](https://www.youtube.com/watch?v=1mXUFweWOug)
- [Kolmogorov-Arnold Networks](https://www.youtube.com/watch?v=lEwTkVNZPAU&list=PLE-clsGWgs9IeW4UGXzZWjRWGy9ZigEBY&index=12)
- [Fractured Entangled Representation Hypothesis](https://github.com/akarshkumar0101/fer)

### Phase 1: Basic KAN

- Train a basic KAN on some picbreedr images, along with standard neural networks
- Mutate specific weight ids to experiment with the network representation of said images (see references)

### Phase 2: Swarm KAN

- Implement a swarm-based KAN to replace spline functions
- Same A/B testing vs Basic KAN and Classic NN on image weight relationships

### Phase 3: Memetic KAN

- Use a memetic algorithm to jointly evolve weights and activation functions (instead of keeping eveyrthing at 1 like in a KAN)
- Train on the weight modified images of skulls, apples, etc. and see if we can derive the weight-agnostic i.e. "pure" image from this training
    - From a fat skull, can we train a network to somehow represent the "normal" skull (constant coefficients) by resetting the weights post-training?
- Same A/B test