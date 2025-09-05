# 🚀 Previsor de Matriz V-Cov com RNN

Sistema avançado para previsão de matrizes de variância-covariância usando decomposição de Cholesky e redes neurais LSTM.

## 📁 Estrutura do Projeto

```
📦 Projeto/
├── 🐍 main.py                 # Ponto de entrada principal
├── 🧠 vcov_predictor.py       # Lógica de machine learning e processamento
├── 🎨 gradio_interface.py     # Interface gráfica web (Gradio)
└── 📄 README.md               # Este arquivo
```

## 🏗️ Arquitetura Modular

### 📄 `main.py`
- **Função**: Ponto de entrada da aplicação
- **Responsabilidade**: Inicializar e lançar a interface Gradio
- **Tamanho**: ~40 linhas (clean & minimal)

### 🧠 `vcov_predictor.py`
- **Função**: Motor de machine learning
- **Responsabilidades**:
  - Download de dados via Yahoo Finance
  - Cálculo de matrizes de variância-covariância
  - Decomposição de Cholesky
  - Treinamento de modelo LSTM
  - Previsões quantitativas
- **Tamanho**: ~300 linhas (core business logic)

### 🎨 `gradio_interface.py`
- **Função**: Interface gráfica web
- **Responsabilidades**:
  - Criação da interface Gradio
  - Visualizações com Plotly
  - Gerenciamento de eventos UI
  - Formatação de resultados
- **Tamanho**: ~200 linhas (presentation layer)

## 🚀 Como Executar

```bash
# 1. Ativar ambiente virtual
source venv/bin/activate

# 2. Instalar dependências (se necessário)
pip install gradio plotly yfinance pandas numpy scipy tensorflow scikit-learn

# 3. Executar aplicação
python main.py
```

## 🌐 Acesso

- **URL Local**: http://localhost:7860
- **Interface**: Web-based via Gradio
- **Compatibilidade**: Todos os navegadores modernos

## 🔧 Vantagens da Modularização

### ✅ **Separação de Responsabilidades**
- **Business Logic** isolada da **UI**
- **Facilita testes** unitários
- **Reutilização** de código

### ✅ **Manutenibilidade**
- **Código limpo** e organizado
- **Fácil debugging** e desenvolvimento
- **Extensibilidade** para novas features

### ✅ **Escalabilidade**
- **Diferentes interfaces** (CLI, API, Desktop)
- **Deploy independente** de componentes
- **Performance** otimizada

## 🛠️ Tecnologias Utilizadas

### **Backend (ML)**
- **TensorFlow/Keras**: Redes neurais LSTM
- **NumPy/SciPy**: Computação científica
- **Pandas**: Manipulação de dados
- **scikit-learn**: Pré-processamento
- **yfinance**: Dados financeiros

### **Frontend (UI)**
- **Gradio**: Interface web moderna
- **Plotly**: Visualizações interativas
- **HTML/CSS**: Estilização customizada

## 📊 Funcionalidades

- **📈 Previsão de Matrizes V-Cov**: Usando LSTM + Cholesky
- **🔥 Heatmaps Interativos**: Visualização de correlações
- **📋 Relatórios Detalhados**: Análise quantitativa completa
- **💡 Exemplos Pré-definidos**: Testes rápidos
- **⚡ Processamento em Tempo Real**: Interface responsiva

## 🎯 Casos de Uso

- **Gestão de Portfólio**: Otimização de alocação
- **Análise de Risco**: Medição de volatilidade
- **Trading Quantitativo**: Estratégias baseadas em IA
- **Pesquisa Acadêmica**: Estudos de correlação

## ⚠️ Disclaimer

Este sistema é para fins **educacionais** e de **pesquisa**. Não constitui aconselhamento financeiro.

---
