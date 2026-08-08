### right now i dont know how to start this project i have initialized some folders but dont know what to do

the thing that is in my mind is to start with

first ill make gpt 2 architecture for that ill write the component of gpt2 in this modules than assemble it model than write code to inference chat alone than with techniques like temparature scaling and top p after that ill make dataset and dataloader to get the training data then ill write the loss and accuracy measures and then training loop

make whatwhere is easy and i know then change it to make it from scratch for example use nn.LinearNorm instead of making one when it works then make LinearNorm

#### things a training function also does

* Generate text every epoch
* Save best checkpoint
* Save last checkpoint
* TensorBoard
* Weights & Biases
* Early stopping
* Email notification (extreme example!)

my approach for model architecture is that ill build the base components like pos embedding token embedding layer norm ffn moe attention like gqa , mqa,mha etc now when i want to create a new architecture i have to make my transformer from these component then arange it in model
