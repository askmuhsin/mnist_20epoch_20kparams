## Working instructions to claude
- we use uv for this project
- with marimo notebooks for development and testing
- along with python scritps for finalized implementations or when its relevant
- no major change sweeps in a single steps
- all individual changes should be small and focused
- I will review the changes and provide feedback
- this is very important; overall verbosity of your writings in this file and elsewhere in chat should be minimal. set verbosity to 3/10. prefer concise and dense reponses over stylized or verbose responses.
- no comments in code, minimal markdown or comments in marimo too.
- dont overdo, if i ask to do EDA provide only the things that really matter, or clarify with me, dont go overboard. This should be the principle throughout.
- you are assisting me in development.
- if i want you to do tasks end to end i will be explicit about it.

### Dependencies
- Core libraries in pyproject.toml: torch, torchvision, pytorch-lightning, marimo, matplotlib, numpy
- mlflow installed globally as uv tool (not in project dependencies)

### Project Structure
- `data/` - datasets (gitignored)
- `experiments/` - experiment outputs (gitignored)  
- `logs/` - training logs (gitignored)
- `outputs/` - model outputs (gitignored)
- `checkpoints/` - model checkpoints (gitignored)


<IMPORTANT: Do not edit or modify the following section of this file. >
# Muhsin's notes on back to mnist 

## in the final state of the project
i want to train an mnist model with a few contstraints 
- need to achieve 99.4% validation/test accuracy
- 50/10k split (so train on 50k and validate on 10k) Note: validation and test is the same
- parameters should be less than 20k
- we have a list of things we can use, which i will go over in the components section, not necessarily all has to be used. 
- Have used BN and Dropout.
- Less than 20 Epochs
- Nice to have - a fully connected layer or have used GAP (Global Average Pooling)
- github actions will show
- - total parameters count
- - use of BN and Dropout
- - use of fully connected layer or GAP
- test accuracy logs need to be saved

### components and other points
- maxpooling, 1x1 conv, 3x3 conv
- softmax, lr
- BN, image normalization
- position of maxpooling
- how many layers, what is the receptive field (in computer vision receptive field means the area of the input that a given neuron in the output layer is sensitive to)
- how the number of kernels are decided
- transition layers (how is it positioned)
- Dropout (when is it introduced, ie.. not the location of it in the network but it is introduced when we start observing the validation/test accuracy shows signs of overfitting)
- the distance of maxpooling from the prediction layer
- the distance of batch normalization from the prediction layer
- when do we stop convolutions and go ahead with a larger kernel or some other alternatives. (not critical at this stage)
- How do we know the training is not going well, comparatively very very early
- batch size and effects of batch size

## breakdown of key phases
- i dont need to read or watch materials until it becomes essential
- 1. load pytorch, and essentials, run some quick code tests
- 2. use pytorch lightning
- 3. check accelerator availability
- 4. download, load dataset
- 5. connect mlflow
- 6. basic model to meet the key constraints
- 7. take it from there.

<IMPORTANT: Do not edit or modify the following section of this file. >