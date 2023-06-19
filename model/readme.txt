o enunciado do trabalho é o seguinte:

A partir de uma base de imagens, a equipe deve escolher e implementar um classificador(rede neural, SVM ou CNNs) utilizando as bibliotecas TensorFlow / Keras/scikit learn. Por último, o seu trabalho deverá ser apresentado para a turma e descrito na forma de relatórioao qual as regras estão descritas abaixo.



Preciso que voce gere no final pelo menos 2 formas de analisar a efetividade do modelo, tempo de execucao, acuracia, recall e graficos e tambem um arquivo .h5 para importar em outro projeto


o meu dataset consiste em uma base de raiox do pulmao para deteccao de pneumonia, a estrutura é seguinte:

tenho uma pasta chamada de input, dentro dela tem outras 3 pastas: test, train e val
dentro de cada uma eles tem a mesma estrutura 2 pastas contendo imagens de pneumonia no fomrato .jpeg e imagens de pulmao normal no formato .jpeg tambem. A pasta test é para testes, a pasta train é para treinar o modelo e a pasta val é para validacao


#pip install --upgrade tensorflow