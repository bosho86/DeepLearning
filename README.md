# DeepLearning
Project Course Winter 2019
#Determination of heart diseases from ECG signals using dense neural networks

#Introduction
Thanks to the machine learning and deep learning libraries developed in the past couple of decades, early detection of the
cardiac abnormalities and the development of automation tools for the diagnosis has seen significant growth in the past 10
years. Despite these results conventional machine learning approaches require lots of time and effort for feature selection [1].
It is necessary to select the best features as input for the supervised classification algorithms which is found by a lot of 
trials, and might be time consuming. Advantage of deep learning over traditional machine learning is, that it doesn’t need 
feature extraction and feature selection. The decision-making algorithm can consider all available evidence. 
Even though the theory and the applications of deep learning are still evolving, several published [3-6] studies have shown
that it possesses the capability of faster and more reliable diagnoses in physiological signals. In some case, the deep 
learning architectures have shown their usefulness by surpassing the performance of traditional supervised classification
machine learning techniques [5]. This predicts a trend, where we might shift away from the currently used decision support
methods, such as (SVM) and K-Nearest Neighbour (K-NN), towards deep learning methods [4]. For instance, Acharya et al. [4] 
reported an accuracy of 98%. Furthermore, their system outperformed traditional approaches and it has the added benefit of
not having to perform feature extraction and de-noising of the signals. 
Currently application of deep learning to physiological signals uses Convolutional Neural Networks and Autoencoders [1,4]. 
The objective of this work is to use deep learning methods (CNN) applied to ECG signals to detect heart disorders in the 
measured subjects. The labeled dataset for training the CNN was obtained using Electro-Cardiogram (ECG) measurements, which are
typically performed by placing electrodes on the chest [1]. This signal reflects the functioning of the heart and it has well 
distinguishable features. However, the difficulty in ECG interpretation is to spot the morphological changes which indicate a
particular cardiac problem or diabetes [2]. These abnormalities may be minute and very often they may be transients or present 
all the time. A supervised learning based approach may be useful to automatically detect such abnormalities in the ECG.

The report is organized as follows. The information about training dataset, the preprocessing techniques, and the architecture
of the deep neural network used in this work are described in Section II.
The results of the training dataset and of optimization of some of the hyper-parameters are discussed in Section II. Finally, 
Section IV concludes the report.

