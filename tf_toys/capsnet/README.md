# Capsule network TF2
## The purpose of this folder is to reproduce and practice well defined Capsule Networks
* The first target is Dynamic Routing, EM Routing, Self-Routing since the Self-Routing authors already published well organized example source code in PyTorch 1.2. 

|               | Dynamic Routing [1]          | EMRouting [2]                | Self-Routing[3]                                                                     | Inverted Dot-Product Attention Routing [4] |
|---------------|------------------------------|------------------------------|-------------------------------------------------------------------------------------|--------------------------------------------|
| Routing       | sequential iterative routing | sequential iterative routing | non-iterative routing                                                               | concurrent iterative routing               |
| Poses         | vector                       | matrix                       | matrix                                                                              | matrix                                     |
| Activations   | n/a (norm poses)             | determined by EM             | the summation of the weighted votes of lower_level capsules over spatial demensions | n/a                                        |
| Non-linearity | Squash function              | n/a                          | n/a                                                                                 | n/a                                        |
| Normalization | n/a                          | n/a                          | n/a                                                                                 | Layer Normalization                        |
| Loss Function | Margin Loss                  | Spread loss                  | I'll check                                                                        | Cross Entropy / Binary Cross Entropy       |
* Please refer to the original version of this table in the section 4.3 of [4], I just added a Self-Routing column. 

## Credit
- https://github.com/ageron/handson-ml, capsnet_tutorial.py
- https://github.com/coder3000/SR-CapsNet, all other source scripts.

## Reference
[1] Sara Sabour, Nicholas Frosst, Geoffrey E. Hinton, Dynamic Routing Between Capsules, NIPS 2017  
[2] Hinton, G., Sabour, S., Frosst, N., 2018. Matrix capsules with em routing. In: ICLR, pp. 1–15. https://doi.org/10.2514/6.2003-4412  
[3] Taeyoung Hahn, Myeongjang Pyeon, Gunhee Kim, Self-Routing Capsule Networks, NIPS 2019
[4] Yao-Hung Hubert Tsai, Nitish Srivastava, Hanlin Goh, Ruslan Salakhutdinov: Capsules with Inverted Dot-Product Attention Routing. ICLR 2020
