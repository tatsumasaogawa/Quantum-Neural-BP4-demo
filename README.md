# Neural Belief Propagation Decoding of Quantum LDPC Codes Using Overcomplete Check Matrices
## Sisi Miao, Alexander Schnerring, Haizheng Li, and Laurent Schmalen

This repository provides an implementation that produces the results presented in [1]. It consists of two main core parts: One is the Jupyter Notebook which implements the neural BP network. Another is the C++ codes which evaluates the performance of the trained decoder.

Specifically, the Jupyter Notebook contains:
* The nerual BP4 decoder, which is defined as the class 'NBP_oc'
* Functions for training, which can be run on a CPU or GPU
* Calls to C++ functions that evaluate the trained decoder
* Overcomplete check matrices used in [1], which are provided in the ./PCMs folder

The C++ codes contains:
* 'simulateFER', which calls the decoder to simulate random errors with different depolarizing probabilities. The output is the depolarizing probability versus frame error rate.
* 'stabilizerCodes', which is the class that performs the decoding, including adding random errors, performing belief propagation, and checking for decoding success.
* 'fileReader', which is the helper class that reads in and stores the check matrices, as well as the trained weights generated by the Jupyter Notebook. The matrices are then passed to the instance of the stabilizerCodes class.


[1] S. Miao, A. Schnerring, H. Li and L. Schmalen, "Neural belief propagation decoding of quantum LDPC codes using overcomplete check matrices," Proc. IEEE Inform. Theory Workshop (ITW), Saint-Malo, France, Apr. 2023, https://arxiv.org/abs/2212.10245

## Abstract
The recent success in constructing asymptotically good quantum low-density parity-check (QLDPC) codes makes this family of codes a promising candidate for error-correcting schemes in quantum computing. However, conventional belief propagation (BP) decoding of QLDPC codes does not yield satisfying performance due to the presence of unavoidable short cycles in their Tanner graph and the special degeneracy phenomenon. In this work, we propose to decode QLDPC codes based on a check matrix with redundant rows, generated from linear combinations of the rows in the original check matrix. This approach yields a significant improvement in decoding performance with the additional advantage of very low decoding latency. Furthermore, we propose a novel neural belief propagation decoder based on the quaternary BP decoder of QLDPC codes which leads to further decoding performance improvements.

## 
<sub> Sisi Miao would like to thank Marcus Müller for improving the quality of the codes.</sub>
