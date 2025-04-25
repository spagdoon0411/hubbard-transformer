# Model Implementation Notes

## Research Questions

- Should we have site embeddings and global sequence embeddings?
- Should the de-embedding function take all logits to the next logit or all
  logits to the next logit? What's the point of strictly using the last one?
- Proposition: input token embeddings have to be completely homogeneous for
  autoregressive sampling to really work.
- Does it benefit us to impose some ordinal interpretation on occupation
  numbers? Or might it be better to treat them as purely categorical?
- Think about how to abstract symmetries as a class, etc.? And of how to apply
  symmetries to the model. What systems might this be useful for?
- What paradigm is best: one-hot, multi-one-hot, or tensor containing integers?
  - Tensor containing integers is bad when you have to produce a probability
    distribution over the possible next token
- Does it make physical sense to impose a maximum number of particles occupying
  each state?
- How can we support branching coverage or most-probable sampling?
  - The Ising network populates a complete binary tree (b=2); we can't do this
    because for us b=large
- Compare sampling multiple token branches from the same next-logit
  distribution
- Should we make tokens one-hot or indices? Should we constrain to one-hot only
  in the transformation made by the output heads?
- How might we investigate the effect of sampling more closely?
- What do the phases belong to? Maybe we should just map each per-site logit to
  a phase directly?
- Consider adding a loss term for the particle number
- Should the particle number equalizer node be completely stochastic without
  any model parameter influence?
- Conflict between prob generation of occupation numbers and particle number
  equalization
- Does the sampling node have a zero gradient with respect to model params?
- Can we force a focus on certain chain stubs?
- Should the model output log probs or probs?

  - Argument for probs: we sample several times for a single computation of our
    loss function

- We've already calculated the probabilities associated with each site; is it
  ineefficient to re-embed and re-calculate on the loss function calculation
  step?

  - This depends on whether the embeddings change

- Is embedding parallelizable across the sequence dimension? Can we just retain
  the probs in mem and use them in the calculation of e_loc?
- This also means we should calculate the phases during sampling and save them
  for loss calculation
- NOTE: in the most general case embeddings are not parallelizable over
  sequence dimension: embedding module can do anything it wants with the input
  tokens
- Can we use logits instead of probabilities everywhere? (eventually that
  calculation must happen--or does this sitestep some prob normalization?)
- In the double sum for the hopping term, do we need index information from
  either sum to compute the sign?
- Can we use some proxy function in place of E_loc?
- Is this even feasible with the amount of compute resources we have?

## TODOs

- Better descriptions of each layer
- Be double-sure that the phase head in the de-embedding didn't cause
  cross-batch interactions
- Think harder about what effect rearranging the multidimensional input tokens
  might have on the de-embedding, how the embedding logits pass through the
  model, etc.
- Do row-vs-column major optimizations
- Linear map to reshape in the de-embedding layer felt vulgar
- Is embedding totally parallelizable over the sequence dimension? Verify this
  empirically
- The phase head has a singular output per site;
- How much memory overhead does this sampling strategy add?
