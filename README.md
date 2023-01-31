

# Introdução
Este projeto, denominado de 'where's my coffe mug?', tem como meta principal a deteção de objetos colocados em cima de uma
mesa presente numa cena.
 Para um melhor discernimento do seguimento das tarefas e funções deste programa, o mesmo foi separado por objetivos:


# Objetivo 1 - Treino de um classificador em deep learning
Utilizando o 'RGB-D Object Dataset' desenvolveu-se uma rede de deep learning capaz de efetuar a classificação dos objetos. Esta rede dividiu o dataset em treino e teste (80% / 20%).
A pasta Model Training contém todos os scripts necessarios para a correcta execução do 'model_training.py', que é o script no qual é possivel efetivamente treinar um modelo em deep learning, este programa carrega um conjunto de dados RGB-D e interrompe o processo de treino se o numero máximo de epocas for atingido ou o limite da loss for alcansado.
Este programa também fornece um relatório visual do desempenho do modelo nos conjuntos de treino e teste. As imagens 1 e 2 são ilucidativas daquilo que foi em cima explicado.
![Imagem 1](../SAVITP2/SAVI%20TP2/Readme%20Images/1.png)
![Imagem 2](../SAVITP2/SAVI%20TP2/Readme%20Images/2.png)

 O 'classification_visualizer.py' implementa uma classe que é usada para visualizar os resultados da classificação de imagens. Tem em 3 argumentos, onde as entradas são as imagens, os rótulos são a verdade e as saídas são as previsões do modelo.
 O 'data_visualizer.py'é uma classe que capaz de representar pontos na janela com um grafico carteziano com determinados parâmetros .
 
A classe Dataset recebe uma lista de nomes de arquivos de imagem como entrada e posteriormente cria rótulos para cada imagem com base no nome do arquivo e armazena-os na lista self.labels. A classe também define um conjunto de transformações que são aplicadas a cada imagem.
 
 O script 'model.py' implementação de um modelo de Rede Neural para classificação de imagens. A primeira camada totalmente conectada mapeia os recursos extraídos para um espaço de 10 dimensões e a segunda camada totalmente conectada mapeia para um espaço de 51 dimensões. O modelo também possui uma camada de dropout para evitar o overfitting.
 
# Objetivo 2 - Pre-processamento 3D



# Objetivo 3 - Classificação de objetos na cena



# Objetivo 4 - Descrição áudio da cena


# Objetivo 5 - Sistema em tempo real