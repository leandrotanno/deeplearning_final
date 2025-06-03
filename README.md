# Classificador de Imagens com CNN no CIFAR-10 🧠📷

Este projeto utiliza uma rede neural convolucional (CNN) para classificar imagens do conjunto de dados CIFAR-10. O modelo é implementado com TensorFlow/Keras e usa técnicas de *data augmentation*, *batch normalization* e *early stopping* para melhorar a performance.

## 🔍 Descrição

O CIFAR-10 é um dataset com 60.000 imagens coloridas de 32x32 pixels divididas em 10 classes (avião, carro, gato, cachorro etc). Este projeto treina uma CNN para reconhecer essas categorias.

### Principais etapas do projeto:

- Carregamento e normalização do dataset
- Visualização de amostras de imagens
- Data Augmentation com `ImageDataGenerator`
- Definição de arquitetura CNN com:
  - Camadas `Conv2D`, `BatchNormalization`, `MaxPooling2D`, `Dropout`
- Treinamento com callbacks:
  - `EarlyStopping`
  - `ReduceLROnPlateau`
- Avaliação final no conjunto de teste
- Visualização da acurácia por época
- Predições visuais (acertos e erros coloridos)

## 🚀 Como executar

1. Clone o repositório:

```bash
git clone https://github.com/leandrotanno/deeplearning_final
cd cnn-cifar10
```

2. Crie um ambiente virtual (opcional, mas recomendado):

```bash
python -m venv venv
source venv/bin/activate  # Linux/Mac
venv\Scripts\activate   # Windows
```

3. Instale as dependências:

```bash
pip install -r requirements.txt
```

4. Execute o script:

```bash
python main.py
```

## 📈 Resultados

- Acurácia na base de teste: aproximadamente **80%**
- Visualizações claras de desempenho e predições corretas/incorretas.

## 📁 Estrutura do Projeto

```
deeplearning_final/
├── main.py
├── requirements.txt
└── README.md
```

## 🧠 Tecnologias Utilizadas

- Python
- TensorFlow / Keras
- Matplotlib
- NumPy

## 📸 Exemplos

- Visualização das imagens de treino
- Curva de acurácia
- Exemplos de predições com cores indicando acertos (verde) e erros (vermelho)

