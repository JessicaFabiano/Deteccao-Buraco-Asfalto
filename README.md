# Deteção de Buracos em Asfalto com YOLOv8

Este projeto consiste em um sistema de Visão Computacional, capaz de identificar e demarcar buracos em vias públicas com **94% de precisão**. O objetivo é criar uma ferramenta automatizada para auxiliar na manutenção viária.

![Exemplo de Deteção](https://raw.githubusercontent.com/JessicaFabiano/Deteccao-Buraco-Asfalto/refs/heads/main/teste2.png)

# O Desafio
Durante o desenvolvimento, perebi um problema de **Data Leakage (Vazamento de Dados)**. Inicialmente, o modelo apresentava uma precisão de 99%. Ao investigar, descobri que o dataset não tinha uma separação física adequada entre treino e validação, fazendo com que a IA "decorasse" as respostas em vez de aprender padrões.

Para garantir a robustez do modelo, desenvolvi um **script de pré-processamento em Python** que:
- Verifica automaticamente a distribuição das pastas.
- Se a validação for insuficiente, o script isola aleatoriamente 15% das imagens (`random.shuffle`) antes do início do treino.
- Reescreve o arquivo de configuração `data.yaml` para garantir que o modelo aponte para os diretórios corretos.

Essa abordagem reduziu o "falso positivo" de 99% para uma métrica real e robusta de **94.4%**, garantindo que o modelo funciona em imagens nunca vistas antes.

# Resultados
O modelo final foi treinado utilizando **Transfer Learning** (baseado no YOLOv8 Nano) por 25 épocas.

**Precisão (mAP50):** 94.4% / **Tempo de Inferência:** Real-time (adequado para vídeo)

![Gráfico de Treino](https://raw.githubusercontent.com/JessicaFabiano/Deteccao-Buraco-Asfalto/refs/heads/main/gr%C3%A1fico.png)

# Como Executar
O projeto foi desenvolvido para rodar em nuvem (Google Colab) ou localmente.

# Pré-requisitos
* Python 3.x
* Ultralytics (`pip install ultralytics`)
* Roboflow (para gestão do dataset)

# Código de Inferência Simples
```python
from ultralytics import YOLO

# Carregar o modelo treinado
model = YOLO('best.pt')

# Realizar deteção numa imagem nova
model.predict('foto_da_rua.jpg', show=True, conf=0.3)



